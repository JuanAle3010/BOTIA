import torch
import torch.nn as nn
import torch.nn.functional as F


class NessieEvaluatorNet(nn.Module):
    """
    Arquitectura neuronal inspirada en Nessie123 (~1.5M de parámetros).
    Procesa 13 tokens de estado (1 Campo + 12 Pokémon OTS en Reg M-C) y emite:
    - Distribución de valor (para valor esperado y cálculo exacto de entropía de hoja)
    - Logits de política conjunta para inicializar el solver Double-Oracle
    - Cabezas auxiliares de predicción de estado futuro para máxima eficiencia muestral
    """
    def __init__(
        self,
        token_input_dim: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_value_bins: int = 21,
        num_joint_actions: int = 90
    ):
        super().__init__()
        self.num_value_bins = num_value_bins

        # Proyección de entrada para los 13 tokens (1 Field + 6 P1 Mons + 6 P2 Mons)
        self.input_projection = nn.Sequential(
            nn.Linear(token_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Embeddings posicionales por rol (Campo, Activo 1, Activo 2, Reserva 1-2, No elegidos)
        self.role_embedding = nn.Embedding(13, d_model)

        # Tronco Transformer Encoder (~1.2M parámetros con d_model=256, 3 capas)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 1. Cabeza de Valor Distribuido (Categorial sobre 21 bins de [0.0 a 1.0] winrate)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_value_bins)
        )

        # 2. Cabezas de Política Prioritaria para P1 y P2 (Guía de búsqueda)
        self.policy_p1_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_joint_actions)
        )
        self.policy_p2_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_joint_actions)
        )

        # 3. Cabezas Auxiliares (Entrenamiento de alta eficiencia muestral en 375k juegos)
        self.aux_survival_head = nn.Linear(d_model, 1)
        self.aux_next_turn_ko_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )
        self.aux_future_hp_diff = nn.Linear(d_model, 1)

        self.register_buffer("value_support", torch.linspace(0.0, 1.0, num_value_bins))

    def forward(self, state_tokens: torch.Tensor, return_aux: bool = False):
        """
        state_tokens: Tensor de forma (Batch, 13, token_input_dim)
        """
        batch_size = state_tokens.size(0)
        positions = torch.arange(13, device=state_tokens.device).unsqueeze(0).expand(batch_size, -1)

        x = self.input_projection(state_tokens) + self.role_embedding(positions)
        encoded = self.transformer(x)

        global_repr = 0.5 * encoded[:, 0, :] + 0.5 * encoded[:, 1:, :].mean(dim=1)

        value_logits = self.value_head(global_repr)
        value_probs = F.softmax(value_logits, dim=-1)
        expected_winrate = torch.sum(value_probs * self.value_support, dim=-1)

        # Entropía de Shannon de la evaluación estática (clave en Nessie123)
        leaf_entropy = -torch.sum(value_probs * torch.log(value_probs + 1e-8), dim=-1)

        p1_logits = self.policy_p1_head(global_repr)
        p2_logits = self.policy_p2_head(global_repr)

        output = {
            "winrate": expected_winrate,
            "leaf_entropy": leaf_entropy,
            "value_logits": value_logits,
            "p1_logits": p1_logits,
            "p2_logits": p2_logits
        }

        if return_aux:
            mon_tokens = encoded[:, 1:, :]
            output["aux_survival_logits"] = self.aux_survival_head(mon_tokens).squeeze(-1)
            output["aux_ko_logits"] = self.aux_next_turn_ko_head(global_repr)
            output["aux_hp_diff"] = torch.tanh(self.aux_future_hp_diff(global_repr)).squeeze(-1)

        return output
