import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple, Dict, Callable


def solve_zero_sum_matrix_game(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Resuelve el equilibrio de Nash exacto de una matriz de pagos U (m x n) en [0, 1]
    mediante Programación Lineal (Método HiGHS de SciPy).
    Retorna: (estrategia_p1, estrategia_p2, valor_del_juego)
    """
    m, n = payoff_matrix.shape
    if m == 1 and n == 1:
        return np.array([1.0]), np.array([1.0]), float(payoff_matrix[0, 0])

    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0) for _ in range(m)] + [(None, None)]

    res_p1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res_p1.success:
        return np.ones(m) / m, np.ones(n) / n, float(np.mean(payoff_matrix))

    x_star = np.clip(res_p1.x[:m], 0.0, 1.0)
    x_star /= np.sum(x_star)
    game_value = float(res_p1.x[-1])

    c_p2 = np.zeros(n + 1)
    c_p2[-1] = 1.0
    A_ub_p2 = np.hstack([payoff_matrix, -np.ones((m, 1))])
    b_ub_p2 = np.zeros(m)
    A_eq_p2 = np.zeros((1, n + 1))
    A_eq_p2[0, :n] = 1.0
    bounds_p2 = [(0.0, 1.0) for _ in range(n)] + [(None, None)]

    res_p2 = linprog(c_p2, A_ub=A_ub_p2, b_ub=b_ub_p2, A_eq=A_eq_p2, b_eq=b_eq, bounds=bounds_p2, method="highs")
    if res_p2.success:
        y_star = np.clip(res_p2.x[:n], 0.0, 1.0)
        y_star /= np.sum(y_star)
    else:
        y_star = np.ones(n) / n

    return x_star, y_star, game_value


class Bo3OpponentTracker:
    """
    Mejora sobre Nessie123: Rastreador Bayesiano de tendencias del rival a lo largo de un set Bo3.
    Registra frecuencias de uso de Protect, ataques en área, cambios y control de velocidad.
    """
    def __init__(self, prior_strength: float = 4.0):
        self.prior_strength = prior_strength
        self.action_category_counts = {
            "protect_turn": 1.0,
            "spread_attack": 1.0,
            "single_target": 1.0,
            "switch_action": 1.0,
            "speed_control": 1.0
        }

    def record_observed_opponent_action(self, category: str):
        if category in self.action_category_counts:
            self.action_category_counts[category] += 1.0

    def adjust_opponent_strategy(self, nash_y: np.ndarray, action_categories: List[str], exploit_weight: float = 0.25) -> np.ndarray:
        total = sum(self.action_category_counts.values())
        empirical_weights = np.array([self.action_category_counts.get(cat, 1.0) / total for cat in action_categories])
        if np.sum(empirical_weights) > 0:
            empirical_weights /= np.sum(empirical_weights)
        else:
            return nash_y

        blended_y = (1.0 - exploit_weight) * nash_y + exploit_weight * empirical_weights
        return blended_y / np.sum(blended_y)


class EntropyGuidedDoubleOracleEngine:
    """
    Motor de búsqueda principal de Nessie123 mejorado:
    1. Resuelve la matriz simultánea en el nodo raíz mediante Double-Oracle.
    2. Calcula la entropía estática de cada hoja (leaf_entropy).
    3. Profundiza selectivamente las hojas con mayor (Probabilidad de Equilibrio * Entropía).
    """
    def __init__(self, evaluator_fn: Callable, simulator_step_fn: Callable, bo3_tracker: Bo3OpponentTracker = None):
        self.evaluator_fn = evaluator_fn
        self.simulator_step_fn = simulator_step_fn
        self.bo3_tracker = bo3_tracker or Bo3OpponentTracker()

    def search(
        self,
        root_state,
        p1_actions: List,
        p2_actions: List,
        p2_categories: List[str],
        inference_budget_evals: int = 250,
        max_seeds_per_leaf: int = 64
    ) -> Tuple[np.ndarray, float, Dict]:
        m, n = len(p1_actions), len(p2_actions)

        payoff_sum = np.zeros((m, n), dtype=np.float64)
        entropy_matrix = np.ones((m, n), dtype=np.float64) * 1.5
        visit_counts = np.zeros((m, n), dtype=np.int32)

        init_m, init_n = min(m, 6), min(n, 6)
        evals_used = 0

        for i in range(init_m):
            for j in range(init_n):
                next_state, is_terminal, term_val = self.simulator_step_fn(
                    root_state, p1_actions[i], p2_actions[j], seed=0
                )
                if is_terminal:
                    val, ent = term_val, 0.0
                else:
                    val, ent = self.evaluator_fn(next_state)
                payoff_sum[i, j] += val
                entropy_matrix[i, j] = ent
                visit_counts[i, j] = 1
                evals_used += 1

        while evals_used < inference_budget_evals:
            avg_payoff = np.divide(
                payoff_sum, np.maximum(visit_counts, 1),
                out=np.full((m, n), 0.5), where=visit_counts > 0
            )

            x_star, y_star, root_value = solve_zero_sum_matrix_game(avg_payoff)
            y_adjusted = self.bo3_tracker.adjust_opponent_strategy(y_star, p2_categories)

            prob_mass = np.outer(0.85 * x_star + 0.15 / m, 0.85 * y_adjusted + 0.15 / n)
            deepening_priority = (prob_mass * (entropy_matrix + 0.05)) / np.sqrt(1.0 + visit_counts)

            deepening_priority[visit_counts >= max_seeds_per_leaf] = -1.0
            deepening_priority[entropy_matrix <= 1e-5] = -1.0

            if np.max(deepening_priority) <= 0.0:
                break

            best_idx = np.unravel_index(np.argmax(deepening_priority), (m, n))
            i_sel, j_sel = best_idx[0], best_idx[1]
            seed_id = int(visit_counts[i_sel, j_sel])

            child_state, is_term, term_v = self.simulator_step_fn(
                root_state, p1_actions[i_sel], p2_actions[j_sel], seed=seed_id
            )

            if is_term:
                leaf_val, leaf_ent = term_v, 0.0
            else:
                leaf_val, leaf_ent = self.evaluator_fn(child_state)

            payoff_sum[i_sel, j_sel] += leaf_val
            entropy_matrix[i_sel, j_sel] = 0.7 * entropy_matrix[i_sel, j_sel] + 0.3 * leaf_ent
            visit_counts[i_sel, j_sel] += 1
            evals_used += 1

        final_payoff = np.divide(payoff_sum, np.maximum(visit_counts, 1), out=np.full((m, n), 0.5), where=visit_counts > 0)
        x_final, y_final, v_final = solve_zero_sum_matrix_game(final_payoff)

        x_final[x_final < 0.01] = 0.0
        x_final /= np.sum(x_final)

        stats = {
            "support_size": int(np.sum(x_final > 0)),
            "is_pure_strategy": bool(np.max(x_final) >= 0.99),
            "total_evaluations": evals_used,
            "expected_winrate": v_final
        }
        return x_final, v_final, stats
