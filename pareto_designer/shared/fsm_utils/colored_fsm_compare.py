import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from pareto_designer.algorithms.fsm_reduction.colored_db_fsm_reducer import (
    Colored_DB_FSM_Reducer,
)


def cmp_reductions(
    reducer: Colored_DB_FSM_Reducer[str, str, str],
    baseline_reducer: Colored_DB_FSM_Reducer[str, str, str],
    file_path: str,
    reducer_label: str = "target",
    baseline_reducer_label: str = "baseline",
):
    def set_of_state(r: Colored_DB_FSM_Reducer[str, str, str], v: str):
        return set(r.inverse_states_mapping[r.states_mapping[v]])

    def id_of_state(r: Colored_DB_FSM_Reducer[str, str, str], v: str):
        v_in_reduced = r.states_mapping[v]
        out_t = ",".join(r._reduced_fsm.get_outgoing_transitions(v_in_reduced))
        clr = r._reduced_fsm.c(v_in_reduced)
        return f"{v_in_reduced} -> c: {clr}, t: {out_t}"

    def get_approx_binding_score_fn(r: Colored_DB_FSM_Reducer[str, str, str]):
        approx_b_map = r.get_approx_binding_score_map()

        def fn(v: str) -> float:
            v_in_reduced = r.states_mapping[v]
            return approx_b_map[v_in_reduced]

        return fn

    def binding_score(w: str) -> float:
        return reducer.binding_score_map[w]

    b_fn_reducer = get_approx_binding_score_fn(reducer)
    b_fn_baseline_reducer = get_approx_binding_score_fn(baseline_reducer)

    def get_result_for_state(v: str):
        v_id_in_first = id_of_state(reducer, v)
        v_id_in_baseline = id_of_state(baseline_reducer, v)
        v_set_in_first = set_of_state(reducer, v)
        v_set_in_baseline = set_of_state(baseline_reducer, v)

        v_shared = dict()
        v_only_in_first_by_baseline = defaultdict(dict)
        v_only_in_baseline_by_first = defaultdict(dict)

        for w in list(v_set_in_first & v_set_in_baseline):
            v_shared[w] = f"{binding_score(w):.3f}"
        for w in list(v_set_in_first - v_set_in_baseline):
            v_only_in_first_by_baseline[id_of_state(baseline_reducer, w)][
                w
            ] = f"{binding_score(w):.3f}"
        for w in list(v_set_in_baseline - v_set_in_first):
            v_only_in_baseline_by_first[id_of_state(reducer, w)][
                w
            ] = f"{binding_score(w):.3f}"

        return {
            f"id_in_{reducer_label}": v_id_in_first,
            f"score_in_{reducer_label}": f"{b_fn_reducer(v):.3f}",
            f"id_in_{baseline_reducer_label}": v_id_in_baseline,
            f"score_in_{baseline_reducer_label}": f"{b_fn_baseline_reducer(v):.3f}",
            f"only_in_{reducer_label}_by_{baseline_reducer_label}": {
                v: v_only_in_first_by_baseline[v]
                for v in sorted(v_only_in_first_by_baseline.keys())
            },
            f"only_in_{baseline_reducer_label}_by_{reducer_label}": {
                v: v_only_in_baseline_by_first[v]
                for v in sorted(v_only_in_baseline_by_first.keys())
            },
            "shared": {v: v_shared[v] for v in sorted(v_shared.keys())},
        }

    try:
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(get_result_for_state, v) for v in reducer.origin_fsm.V
            ]
            rets = [fut.result() for fut in futures]
            results = dict(zip(reducer.origin_fsm.V, rets))

        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Comparison results (JSON) have been written to '{file_path}'")

    except IOError as e:
        logger.error(f"Error writing to file '{file_path}': {e}")
    except Exception as e:
        logger.exception(str(e))
