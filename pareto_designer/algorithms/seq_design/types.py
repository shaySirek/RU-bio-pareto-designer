from typing import Callable, Iterator, Any

from pareto_designer.algorithms.fsm import T_TRANS

T_BACK_PTR = tuple[T_TRANS, int]
T_SOLUTION = tuple[float, float]
T_SOL_WITH_TRACK = tuple[T_SOLUTION, Any]
T_LAZY_SOL_ITER_FACTORY = Callable[[], Iterator[T_SOL_WITH_TRACK]]
CompareFunc = Callable[[T_SOLUTION, T_SOLUTION], bool]
