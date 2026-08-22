import numpy as np

from pareto_designer.algorithms.seq_design.sampling import SamplingMethod, NO_SAMPLING

PTR_DTYPE = np.dtype([("u_idx", np.uint32), ("sigma_idx", np.uint32), ("j", np.uint32)])
SCORE_DTYPE = np.dtype([("f", "f8"), ("b", "f8")])
EMPTY_PTRS = np.empty(0, dtype=PTR_DTYPE)


def packed_to_ptrs(packed: np.ndarray) -> np.ndarray:
    if packed.size == 0:
        return EMPTY_PTRS
    if packed.ndim == 1:
        packed = packed.reshape(1, 3)
    out = np.empty(len(packed), dtype=PTR_DTYPE)
    out["u_idx"] = packed[:, 0]
    out["sigma_idx"] = packed[:, 1]
    out["j"] = packed[:, 2]
    return out


def ptrs_to_packed(ptrs: np.ndarray) -> np.ndarray:
    packed = np.empty((len(ptrs), 3), dtype=np.uint32)
    packed[:, 0] = ptrs["u_idx"]
    packed[:, 1] = ptrs["sigma_idx"]
    packed[:, 2] = ptrs["j"]
    return packed


def as_ptr_array(pts: np.ndarray | list) -> np.ndarray:
    if isinstance(pts, np.ndarray) and pts.dtype == PTR_DTYPE:
        return pts
    return packed_to_ptrs(np.asarray(pts, dtype=np.uint32))


def find_po_from_arrays(
    f: np.ndarray,
    b: np.ndarray,
    u_idx: np.ndarray,
    sigma_idx: np.ndarray,
    j: np.ndarray,
    sampler: SamplingMethod = NO_SAMPLING,
    position: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Pareto filter + sample on already-offset score arrays.

    Sort by ``f`` descending then ``b`` ascending, keep a point if its
    ``b`` is strictly less than the running minimum (first point always
    kept), group exact ``(f, b)`` ties, then sample.
    """
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=SCORE_DTYPE), []

    order = np.lexsort((b, -f))
    f = f[order]
    b = b[order]
    ptrs = np.empty(n, dtype=PTR_DTYPE)
    ptrs["u_idx"] = u_idx[order]
    ptrs["sigma_idx"] = sigma_idx[order]
    ptrs["j"] = j[order]

    is_po = np.empty(n, dtype=bool)
    is_po[0] = True
    if n > 1:
        is_po[1:] = b[1:] < np.minimum.accumulate(b)[:-1]

    new_run = np.ones(n, dtype=bool)
    if n > 1:
        new_run[1:] = (f[1:] != f[:-1]) | (b[1:] != b[:-1])
    run_starts = np.flatnonzero(new_run)
    n_runs = len(run_starts)
    run_ends = np.empty(n_runs, dtype=np.intp)
    run_ends[:-1] = run_starts[1:]
    run_ends[-1] = n

    po_run = is_po[run_starts]
    po_starts = run_starts[po_run]
    po_ends = run_ends[po_run]
    n_po = len(po_starts)

    po_arr = np.empty((n_po, 2), dtype=np.float64)
    po_arr[:, 0] = f[po_starts]
    po_arr[:, 1] = b[po_starts]
    ptr_groups = [ptrs[po_starts[r] : po_ends[r]] for r in range(n_po)]

    sampled = sampler(po_arr, np.arange(n_po), position)
    key_to_i = {(po_arr[i, 0].item(), po_arr[i, 1].item()): i for i in range(n_po)}
    sel = [key_to_i[(row[0].item(), row[1].item())] for row in np.atleast_2d(sampled)]

    out = np.empty(len(sel), dtype=SCORE_DTYPE)
    out["f"] = po_arr[sel, 0]
    out["b"] = po_arr[sel, 1]
    return out, [ptr_groups[i] for i in sel]
