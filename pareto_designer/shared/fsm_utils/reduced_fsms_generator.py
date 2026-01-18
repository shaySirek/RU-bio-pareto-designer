from typing import Generator

from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)


def get_reduced_fsms(
    fsm_reducer: DB_FSM_Reducer[str, str],
    delta_mse_threshold: float = 5e-3,
    reduction_ratio_threshold: float = 0.1,
) -> Generator[tuple[FSM, dict[str, float], float], None, None]:
    origin_size = len(fsm_reducer.origin_fsm.V)
    previous_fsm = (origin_size, 0.0)
    candidate_reduced_fsm = (fsm_reducer.origin_fsm, fsm_reducer.binding_score_map, 0.0)

    for i, (reduced_fsm, mse, (reduced_fsm_binding_score_map, _, _)) in enumerate(
        fsm_reducer.find_reduced_fsms()
    ):
        if i > 0:
            previous_fsm_n_states, previous_fsm_mse = previous_fsm
            delta_mse = mse - previous_fsm_mse
            reduction_ratio = 1 - (len(reduced_fsm.V) / previous_fsm_n_states)
            if (
                delta_mse > delta_mse_threshold
                and reduction_ratio > reduction_ratio_threshold
            ):
                cndt_fsm, cndt_binding_score_map, cndt_mse = candidate_reduced_fsm
                yield cndt_fsm, cndt_binding_score_map, cndt_mse
                previous_fsm = (len(cndt_fsm.V), cndt_mse)

        candidate_reduced_fsm = (reduced_fsm, reduced_fsm_binding_score_map, mse)
