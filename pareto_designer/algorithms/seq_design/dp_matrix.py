from time import perf_counter
from datetime import datetime
from pathlib import Path
from collections import deque
from shutil import rmtree

import numpy as np
from loguru import logger

from pareto_designer.algorithms.fsm import FSM, T_STATE, T_CHAR
from pareto_designer.algorithms.seq_design.sampling import SamplingMethod, NO_SAMPLING
from pareto_designer.algorithms.seq_design.types import T_SOLUTION, T_BACK_PTR
from pareto_designer.algorithms.seq_design.util import (
    EMPTY_PTRS,
    SCORE_DTYPE,
    as_ptr_array,
    find_po_from_arrays,
    packed_to_ptrs,
    ptrs_to_packed,
)

ITEM_SIZE = 12  # 4 bytes for each index in the 3-tuple
MAX_FILE_SIZE = 16 * (1024**2)  # 16MB
PTR_META_DTYPE = np.dtype(
    [("v_idx", np.int32), ("j", np.int32), ("start", np.int32), ("count", np.int32)]
)


def get_flush_every(fsm: FSM, limit_solutions: int) -> int:
    limit_solutions = limit_solutions or 256
    row_size_in_bytes = len(fsm.V) * limit_solutions * ITEM_SIZE
    return int(MAX_FILE_SIZE / row_size_in_bytes)


class BackPointer:
    __slots__ = ("u_idx", "sigma_idx", "j")

    def __init__(self, u_idx: int, sigma_idx: int, j: int):
        self.u_idx = u_idx
        self.sigma_idx = sigma_idx
        self.j = j


class DP_Matrix:
    def __init__(
        self,
        fsm: FSM,
        n: int,
        flush_every: int,
        checkpoint_dir: str = ".seq_design_dp_checkpoints",
        state_order: list[T_STATE] | None = None,
        alphabet: list[T_CHAR] | None = None,
    ):
        self.fsm = fsm
        self.n = n
        self.flush_every = flush_every
        self.checkpoint_path = (
            Path.home() / checkpoint_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._alphabet = (
            list(alphabet) if alphabet is not None else list(self.fsm.Sigma)
        )
        self._sigma_to_idx: dict[T_CHAR, int] = {
            s: i for i, s in enumerate(self._alphabet)
        }
        states = list(state_order) if state_order is not None else list(self.fsm.V)
        self._state_list = states
        self._state_to_idx: dict[T_STATE, int] = {s: i for i, s in enumerate(states)}
        self._num_states: int = len(self._state_to_idx)
        self._empty_scores = np.empty(0, dtype=SCORE_DTYPE)

        self._back_ptrs_buffer: list[dict[tuple[int, int], np.ndarray]] = []
        self._total_rows_flushed = 0

        self._prev_scores: np.ndarray = np.empty(0, dtype=SCORE_DTYPE)
        self._prev_offsets: np.ndarray = np.zeros(self._num_states + 1, dtype=np.int32)

        self._cached_chunk_idx = -1
        self._cached_chunk_data: np.ndarray | None = None
        self._cached_meta: list[dict[tuple[int, int], tuple[int, int]]] | None = None

    def _get_state_index(self, v: T_STATE) -> int:
        return self._state_to_idx[v]

    def _get_chunk_paths(self, chunk_idx: int) -> tuple[Path, Path]:
        base = self.checkpoint_path / f"chunk_{chunk_idx}"
        return base.with_suffix(".bin"), Path(str(base) + "_meta.npz")

    def _save_chunk(self, chunk_idx: int) -> int:
        bin_path, meta_path = self._get_chunk_paths(chunk_idx)
        start_t = perf_counter()

        ptr_chunks: list[np.ndarray] = []
        meta_chunks: list[np.ndarray] = []
        row_splits = [0]
        total_ptrs = 0
        num_rows = 0
        for row_dict in self._back_ptrs_buffer:
            recs = np.empty(len(row_dict), dtype=PTR_META_DTYPE)
            r = 0
            for (v_idx, j), pts in row_dict.items():
                arr = as_ptr_array(pts)
                recs[r] = (v_idx, j, total_ptrs, len(arr))
                if len(arr):
                    ptr_chunks.append(arr)
                    total_ptrs += len(arr)
                r += 1
            meta_chunks.append(recs)
            row_splits.append(row_splits[-1] + len(recs))
            num_rows += 1

        packed = (
            ptrs_to_packed(np.concatenate(ptr_chunks))
            if ptr_chunks
            else np.empty((0, 3), dtype=np.uint32)
        )
        packed.tofile(bin_path)

        records = (
            np.concatenate(meta_chunks)
            if meta_chunks
            else np.empty(0, dtype=PTR_META_DTYPE)
        )
        np.savez(
            meta_path,
            records=records,
            row_splits=np.asarray(row_splits, dtype=np.int32),
        )

        duration = perf_counter() - start_t
        size_mb = packed.nbytes / (1024 * 1024)
        rate = size_mb / duration if duration > 0 else 0.0
        logger.debug(
            f"Flushed chunk {chunk_idx}: {size_mb:.2f}MB in {duration:.2f}s ({rate:.2f} MB/s)"
        )

        self._back_ptrs_buffer = []
        return num_rows

    def _ensure_chunk_loaded(self, chunk_idx: int) -> None:
        if self._cached_chunk_idx == chunk_idx:
            return
        bin_path, meta_path = self._get_chunk_paths(chunk_idx)
        packed = np.fromfile(bin_path, dtype=np.uint32).reshape(-1, 3)
        self._cached_chunk_data = packed_to_ptrs(packed)
        with np.load(meta_path) as z:
            records = z["records"]
            row_splits = z["row_splits"]
        meta: list[dict[tuple[int, int], tuple[int, int]]] = []
        for r in range(len(row_splits) - 1):
            sl = records[row_splits[r] : row_splits[r + 1]]
            meta.append(
                {
                    (int(x["v_idx"]), int(x["j"])): (int(x["start"]), int(x["count"]))
                    for x in sl
                }
            )
        self._cached_meta = meta
        self._cached_chunk_idx = chunk_idx

    def get_back_ptrs_arr(self, i: int, v_idx: int, j: int) -> np.ndarray:
        if i >= self._total_rows_flushed:
            pts = self._back_ptrs_buffer[i - self._total_rows_flushed].get(
                (v_idx, j), EMPTY_PTRS
            )
            return as_ptr_array(pts)
        chunk_idx = i // self.flush_every
        local_i = i % self.flush_every
        self._ensure_chunk_loaded(chunk_idx)
        loc = self._cached_meta[local_i].get((v_idx, j))
        if loc is None:
            return EMPTY_PTRS
        start, count = loc
        return self._cached_chunk_data[start : start + count]

    def start_row(self):
        self._temp_row_data: list[np.ndarray | None] = [None] * self._num_states
        self._back_ptrs_buffer.append({})

    def update(
        self,
        v: T_STATE,
        scores: list[T_SOLUTION],
        back_ptrs: list[list[T_BACK_PTR]] | None,
    ):
        v_idx = self._get_state_index(v)
        ptr_groups: list[np.ndarray] | None = None
        if back_ptrs:
            ptr_groups = []
            for pts in back_ptrs:
                rows = [
                    (self._get_state_index(u), self._sigma_to_idx[s], jt)
                    for (u, s), jt in pts
                ]
                ptr_groups.append(as_ptr_array(rows))
        self.update_from_arrays(v_idx, np.array(scores, dtype=SCORE_DTYPE), ptr_groups)

    def update_from_arrays(
        self,
        v_idx: int,
        scores: np.ndarray,
        ptr_groups: list[np.ndarray] | None,
    ):
        self._temp_row_data[v_idx] = scores
        if ptr_groups:
            row_dict = self._back_ptrs_buffer[-1]
            for j, pts in enumerate(ptr_groups):
                row_dict[(v_idx, j)] = as_ptr_array(pts)

    def end_row(self, i: int) -> list[int]:
        if not hasattr(self, "_temp_row_data"):
            return

        logger.debug(f"Finished filling row no. {i} / {self.n}")

        sizes = [len(arr) if arr is not None else 0 for arr in self._temp_row_data]
        total = sum(sizes)
        curr_scores = np.empty(total, dtype=SCORE_DTYPE)
        curr_offsets = np.zeros(self._num_states + 1, dtype=np.int32)
        pos = 0
        for v_idx in range(self._num_states):
            curr_offsets[v_idx] = pos
            if self._temp_row_data[v_idx] is not None:
                ln = len(self._temp_row_data[v_idx])
                curr_scores[pos : pos + ln] = self._temp_row_data[v_idx]
                pos += ln
        curr_offsets[self._num_states] = pos
        self._prev_scores, self._prev_offsets = curr_scores, curr_offsets
        del self._temp_row_data

        if len(self._back_ptrs_buffer) >= self.flush_every:
            self._total_rows_flushed += self._save_chunk(
                self._total_rows_flushed // self.flush_every
            )
            logger.info(f"Rows calculated: {i} / {self.n}")

        return sizes

    def get(self, v: T_STATE) -> np.ndarray:
        return self.get_idx(self._get_state_index(v))

    def get_idx(self, v_idx: int) -> np.ndarray:
        s, e = self._prev_offsets[v_idx], self._prev_offsets[v_idx + 1]
        return self._prev_scores[s:e] if s < e else self._empty_scores

    def get_back_ptrs(self, i: int, v_idx: int, j: int) -> list[BackPointer]:
        return [
            BackPointer(int(p["u_idx"]), int(p["sigma_idx"]), int(p["j"]))
            for p in self.get_back_ptrs_arr(i, v_idx, j)
        ]

    def reconstruct_po_set(
        self,
        sampler: SamplingMethod = NO_SAMPLING,
        position: int | None = None,
    ) -> set[tuple[str, T_SOLUTION]]:
        if self._back_ptrs_buffer:
            self._total_rows_flushed += self._save_chunk(
                self._total_rows_flushed // self.flush_every
            )

        position = self.n if position is None else position
        f_parts: list[np.ndarray] = []
        b_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []
        j_parts: list[np.ndarray] = []
        for v_idx in range(self._num_states):
            cell = self.get_idx(v_idx)
            if len(cell) == 0:
                continue
            n_cell = len(cell)
            f_parts.append(cell["f"])
            b_parts.append(cell["b"])
            v_parts.append(np.full(n_cell, v_idx, dtype=np.int32))
            j_parts.append(np.arange(n_cell, dtype=np.int32))

        if not f_parts:
            return set()

        f = np.concatenate(f_parts)
        scores, ptr_groups = find_po_from_arrays(
            f,
            np.concatenate(b_parts),
            np.concatenate(v_parts),
            np.zeros(len(f), dtype=np.int32),
            np.concatenate(j_parts),
            sampler=sampler,
            position=position,
        )

        po_set, queue = set(), deque()
        for score, ptrs in zip(scores, ptr_groups):
            score_t = (float(score["f"]), float(score["b"]))
            for p in ptrs:
                queue.append(("", self.n, int(p["u_idx"]), int(p["j"]), score_t))

        alphabet = self._alphabet
        while queue:
            s, i, v_idx, j, score = queue.popleft()
            if i == 0:
                po_set.add((s, score))
            else:
                for bp in self.get_back_ptrs_arr(i, v_idx, j):
                    queue.append(
                        (
                            alphabet[int(bp["sigma_idx"])] + s,
                            i - 1,
                            int(bp["u_idx"]),
                            int(bp["j"]),
                            score,
                        )
                    )
        return po_set

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rmtree(self.checkpoint_path, ignore_errors=True)
