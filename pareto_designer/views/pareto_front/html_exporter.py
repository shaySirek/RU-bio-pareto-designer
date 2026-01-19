from typing import NamedTuple
from pathlib import Path
from functools import lru_cache

from loguru import logger
from jinja2 import Environment, FileSystemLoader
import numpy as np


class ParetoResult(NamedTuple):
    cost: float
    score: float
    id: str
    url: str
    sequence: str
    target_sequence: str
    costs: np.ndarray


@lru_cache
def _get_env():
    template_dir = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(template_dir))


def render_solution_html(res: ParetoResult, path: Path):
    template = _get_env().get_template("solution.html")
    non_zero = [(int(i), float(res.costs[i])) for i in np.where(res.costs > 0)[0]]
    html_out = template.render(res=res, cost_data=non_zero)
    with (path / res.url).open("w") as f:
        f.write(html_out)


def render_pareto_front(results: list[ParetoResult], motif_id: str, path: Path):
    template = _get_env().get_template("pareto_front.html")
    filename = path / "pareto_front.html"
    html_out = template.render(results=results, motif_id=motif_id)
    with filename.open("w") as f:
        f.write(html_out)

    logger.info(f"Pareto front exported to {filename}")
