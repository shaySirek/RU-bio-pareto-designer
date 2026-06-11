import numpy as np

from pareto_designer.models.context import FSMContext


def get_binding(seq: str, ctx: FSMContext) -> np.ndarray:
    seq_len = len(seq)
    m_len = ctx.motif_length

    data = np.full(seq_len, np.nan)
    if seq_len < m_len:
        return data

    lookup_func = np.vectorize(ctx.binding_score_map.get, otypes=[float])
    valid_len = seq_len - m_len + 1
    windows = [seq[i : i + m_len] for i in range(valid_len)]
    data[:valid_len] = lookup_func(windows)

    return data
