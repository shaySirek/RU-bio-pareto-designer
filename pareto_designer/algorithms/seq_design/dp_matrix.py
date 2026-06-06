import json
from time import perf_counter
from datetime import datetime
from pathlib import Path
from typing import Callable
from collections import deque
import gc
from shutil import rmtree

import numpy as np
from loguru import logger

from pareto_designer.algorithms.fsm import FSM, T_STATE, T_CHAR
from pareto_designer.algorithms.seq_design.types import T_SOLUTION, T_BACK_PTR

SCORE_DTYPE = [("f", "f8"), ("b", "f8")]
ITEM_SIZE = 12  # 4 bytes for each index in the 3-tuple
MAX_FILE_SIZE = 16 * (1024**2)  # 16MB


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
    ):
        self.fsm = fsm
        self.n = n
        self.flush_every = flush_every
        self.checkpoint_path = (
            Path.home() / checkpoint_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._alphabet = list(self.fsm.Sigma)
        self._sigma_to_idx: dict[T_CHAR, int] = {
            s: i for i, s in enumerate(self._alphabet)
        }
        self._state_to_idx: dict[T_STATE, int] = {
            s: i for i, s in enumerate(self.fsm.V)
        }
        self._num_states: int = len(self._state_to_idx)

        self._back_ptrs_buffer: list[dict[tuple[int, int], list[list[int]]]] = []
        self._total_rows_flushed = 0

        self._prev_scores: np.ndarray = np.empty(0, dtype=SCORE_DTYPE)
        self._prev_offsets: np.ndarray = np.zeros(self._num_states + 1, dtype=np.int32)

        self._cached_chunk_idx = -1
        self._cached_chunk_data: np.ndarray | None = None
        self._cached_meta: list[dict[str, tuple[int, int]]] | None = None

    def _get_state_index(self, v: T_STATE) -> int:
        return self._state_to_idx[v]

    def _get_chunk_paths(self, chunk_idx: int) -> tuple[Path, Path]:
        base = self.checkpoint_path / f"chunk_{chunk_idx}"
        return base.with_suffix(".bin"), base.with_suffix(".json")

    @staticmethod
    def _meta_key(v_idx: int, j: int) -> str:
        return f"{v_idx},{j}"

    @staticmethod
    def _build_ptr_list(data: np.ndarray | list[list[int]]) -> list[BackPointer]:
        return [BackPointer(int(p[0]), int(p[1]), int(p[2])) for p in data]

    def _save_chunk(self, chunk_idx: int) -> int:
        bin_path, json_path = self._get_chunk_paths(chunk_idx)
        start_t = perf_counter()

        num_rows = 0
        flat_bps = []
        metadata = []
        for row_dict in self._back_ptrs_buffer:
            row_meta = {}
            for (v_idx, j), pts in row_dict.items():
                start_ptr = len(flat_bps)
                flat_bps.extend(pts)
                row_meta[self._meta_key(v_idx, j)] = (start_ptr, len(pts))
            metadata.append(row_meta)
            num_rows += 1

        final_array = np.array(flat_bps, dtype=np.uint32)
        final_array.tofile(bin_path)
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        duration = perf_counter() - start_t
        size_mb = final_array.nbytes / (1024 * 1024)
        logger.debug(
            f"Flushed chunk {chunk_idx}: {size_mb:.2f}MB in {duration:.2f}s ({size_mb/duration:.2f} MB/s)"
        )

        self._back_ptrs_buffer = []
        del final_array, flat_bps, metadata
        gc.collect()

        return num_rows

    def _load(self, i: int, v_idx: int, j: int) -> list[BackPointer]:
        chunk_idx = i // self.flush_every
        local_i = i % self.flush_every

        if self._cached_chunk_idx != chunk_idx:
            bin_path, json_path = self._get_chunk_paths(chunk_idx)
            self._cached_chunk_data = np.fromfile(bin_path, dtype=np.uint32).reshape(
                -1, 3
            )
            with open(json_path, "r") as f:
                self._cached_meta = json.load(f)
            self._cached_chunk_idx = chunk_idx

        key = self._meta_key(v_idx, j)
        row_meta = self._cached_meta[local_i]
        if key not in row_meta:
            return []

        start, count = row_meta[key]
        slice_data = self._cached_chunk_data[start : start + count]
        return self._build_ptr_list(slice_data)

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
        self._temp_row_data[v_idx] = np.array(scores, dtype=SCORE_DTYPE)
        if back_ptrs:
            row_dict = self._back_ptrs_buffer[-1]
            for j, pts in enumerate(back_ptrs):
                row_dict[(v_idx, j)] = [
                    [self._get_state_index(u), self._sigma_to_idx[s], jt]
                    for (u, s), jt in pts
                ]

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
        v_idx = self._get_state_index(v)
        s, e = self._prev_offsets[v_idx], self._prev_offsets[v_idx + 1]
        return self._prev_scores[s:e] if s < e else np.empty(0, dtype=SCORE_DTYPE)

    def get_back_ptrs(self, i: int, v_idx: int, j: int) -> list[BackPointer]:
        if i >= self._total_rows_flushed:
            pts = self._back_ptrs_buffer[i - self._total_rows_flushed].get(
                (v_idx, j), []
            )
            return self._build_ptr_list(pts)
        return self._load(i, v_idx, j)

    def reconstruct_po_set(self, find_po_func: Callable) -> set[tuple[str, T_SOLUTION]]:
        if self._back_ptrs_buffer:
            self._total_rows_flushed += self._save_chunk(
                self._total_rows_flushed // self.flush_every
            )

        po_scores, tracks = find_po_func(
            lambda u=v: (
                (tuple(sc), (self._get_state_index(u), j))
                for j, sc in enumerate(self.get(v).tolist())
            )
            for v in self.fsm.V
        )

        po_set, queue = set(), deque()
        for score, tracks_list in zip(po_scores, tracks):
            for v_idx, j in tracks_list:
                queue.append(("", self.n, v_idx, j, score))

        while queue:
            s, i, v_idx, j, score = queue.popleft()
            if i == 0:
                po_set.add((s, score))
            else:
                for bp in self.get_back_ptrs(i, v_idx, j):
                    queue.append(
                        (self._alphabet[bp.sigma_idx] + s, i - 1, bp.u_idx, bp.j, score)
                    )
        return po_set

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rmtree(self.checkpoint_path, ignore_errors=True)
