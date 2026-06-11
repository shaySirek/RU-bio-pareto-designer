from pathlib import Path
from functools import lru_cache

from loguru import logger
from jinja2 import Environment, FileSystemLoader

from pareto_designer.models.context import RunContext, ParetoResult


@lru_cache
def _get_env():
    template_dir = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(template_dir))


def render_solution_html(
    ctx: RunContext,
    res: ParetoResult,
    cost_items: list[tuple[int, float, dict | None]],
    seq_with_meta: list[tuple[str, bool]],
):
    template = _get_env().get_template("solution.html")

    html_out = template.render(
        ctx=ctx, res=res, cost_data=cost_items, seq_with_meta=seq_with_meta
    )

    output_file = ctx.output_path / res.url
    with output_file.open("w", encoding="utf-8") as f:
        f.write(html_out)


def render_pareto_front_html(ctx: RunContext, results: list[ParetoResult]):
    template = _get_env().get_template("pareto_front.html")
    filename = ctx.output_path / "index.html"

    html_out = template.render(ctx=ctx, results=results)

    with filename.open("w", encoding="utf-8") as f:
        f.write(html_out)

    logger.info(f"Pareto front exported to {filename}")
