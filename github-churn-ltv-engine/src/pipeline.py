"""One-command orchestration:  python -m src.pipeline [--steps ingest eda train explain ltv score]"""
import argparse
import logging
import time

from src import eda, explain, ingest, ltv, score_customers, train_churn

STEPS = {"ingest": ingest.run, "eda": eda.run, "train": train_churn.run,
         "explain": explain.run, "ltv": ltv.run, "score": score_customers.run}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("pipeline")
    for s in args.steps:
        t = time.time()
        log.info("=== %s ===", s)
        STEPS[s]()
        log.info("=== %s done in %.1fs ===", s, time.time() - t)


if __name__ == "__main__":
    main()
