import argparse

from loguru import logger

from pareto_designer.bio_fetcher.motif import BindingMotif


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_id", type=str, help="Motif matrix ID")
    parser.add_argument("--pattern", "-p", type=str, help="pattern")

    return parser.parse_args()


def main():
    args = parse_args()
    p = args.pattern
    motif = BindingMotif(args.matrix_id)
    logger.info("PWM:\n" + str(motif.pwm))
    logger.info("PSSM:\n" + str(motif.pssm))
    if p and len(p) == motif.length:
        binding_score_map = motif.get_binding_score_map()
        score = binding_score_map.get(p)
        logger.info(f"{p}\t{round(score, 2):.2f}")
