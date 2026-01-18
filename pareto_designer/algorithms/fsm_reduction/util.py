import numpy as np
from loguru import logger

from pareto_designer.algorithms.fsm import T_STATE


def get_scores(
    binding_score_map: dict[T_STATE, float],
    reduced_fsm_binding_score_map: dict[T_STATE, float],
    state_mapping: dict[T_STATE, T_STATE],
) -> tuple[np.ndarray, np.ndarray]:
    n_origin = len(state_mapping)
    origin_scores = np.zeros(n_origin)
    scores_in_reduced = np.zeros(n_origin)

    for i, (v_origin, v_reduced) in enumerate(state_mapping.items()):
        origin_scores[i] = binding_score_map[v_origin]
        scores_in_reduced[i] = reduced_fsm_binding_score_map[v_reduced]

    return origin_scores, scores_in_reduced


def hash_pair(u1: T_STATE, u2: T_STATE):
    return tuple(sorted((u1, u2)))


def get_reduction_efficiency(mse_by_n_states: dict[int, float]) -> float:
    sqrt_n_states = int(np.sqrt(max(mse_by_n_states.keys()) + 1))
    n_states_in_trivial_fsm = min(mse_by_n_states.keys())
    mse_at_sqrt_fsm = mse_by_n_states[sqrt_n_states]
    mse_at_trivial_fsm = mse_by_n_states[n_states_in_trivial_fsm]
    logger.info(
        f"Reduction efficiency is calculated from MSE@sqrt={mse_at_sqrt_fsm:.3f} ({sqrt_n_states} states) "
        f"and from MSE@trivial={mse_at_trivial_fsm:.3f} ({n_states_in_trivial_fsm} state(s))"
    )
    reduction_inefficiency = (
        mse_at_sqrt_fsm / mse_at_trivial_fsm if mse_at_trivial_fsm > 0 else 1
    )
    reduction_efficiency = 1 - reduction_inefficiency
    return reduction_efficiency
