from pathlib import Path
from functools import lru_cache

from loguru import logger
from jinja2 import Environment, FileSystemLoader
import numpy as np

from pareto_designer.models.pareto_front import RunContext, ParetoResult


@lru_cache
def _get_env():
    template_dir = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(template_dir))


def render_solution_html(ctx: RunContext, res: ParetoResult):
    template = _get_env().get_template("solution.html")
    non_zero = [(int(i + 1), float(res.costs[i])) for i in np.where(res.costs > 0)[0]]
    html_out = template.render(ctx=ctx, res=res, cost_data=non_zero)
    with (ctx.output_path / res.url).open("w") as f:
        f.write(html_out)


def render_pareto_front(ctx: RunContext, results: list[ParetoResult]):
    template = _get_env().get_template("pareto_front.html")
    filename = ctx.output_path / "index.html"
    html_out = template.render(ctx=ctx, results=results)
    with filename.open("w") as f:
        f.write(html_out)
    logger.info(f"Pareto front exported to {filename}")
