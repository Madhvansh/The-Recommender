"""Command-line interface for The Recommender.

Subcommands
-----------
- ``train``    : train a model from a YAML config and report test metrics.
- ``evaluate`` : load a checkpoint and evaluate it on the test split.
- ``compare``  : train S4Rec and SASRec under the same config and tabulate the
                 head-to-head deltas.

Examples
--------
    recommender train --config configs/s4rec_amazon_beauty.yaml
    recommender compare --config configs/s4rec_amazon_beauty.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from .pipeline import build_experiment
from .utils.config import Config
from .utils.logging import get_logger
from .utils.seed import set_seed

logger = get_logger("cli")


def _load_cfg(path: str | None) -> Config:
    return Config.from_yaml(path) if path else Config()


def _fmt(metrics: dict) -> str:
    keys = [k for k in metrics if k.startswith(("hit@", "ndcg@", "mrr"))]
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    set_seed(cfg.train.seed)
    exp = build_experiment(cfg, loss_type=args.loss)
    logger.info(
        "dataset=%s users=%d items=%d interactions=%d",
        exp.dataset.name, exp.dataset.num_users, exp.dataset.num_items,
        exp.dataset.num_interactions,
    )
    exp.trainer.fit(exp.train_loader, exp.valid_loader)
    test = exp.trainer.test(exp.test_loader)
    logger.info("TEST | %s", _fmt(test))
    if args.out:
        Path(args.out).write_text(json.dumps(test, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    set_seed(cfg.train.seed)
    exp = build_experiment(cfg, loss_type=args.loss)
    state = torch.load(args.checkpoint, map_location=exp.trainer.device)
    exp.model.load_state_dict(state)
    test = exp.trainer.test(exp.test_loader)
    logger.info("TEST | %s", _fmt(test))


def cmd_compare(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    results: dict[str, dict] = {}
    for name in ("sasrec", "s4rec"):
        run_cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, name=name))
        set_seed(run_cfg.train.seed)
        exp = build_experiment(run_cfg, loss_type=args.loss)
        logger.info("=== training %s ===", name)
        exp.trainer.fit(exp.train_loader, exp.valid_loader)
        results[name] = exp.trainer.test(exp.test_loader)
        logger.info("%s TEST | %s", name, _fmt(results[name]))

    logger.info("=== head-to-head (S4Rec vs SASRec) ===")
    for k in ("hit@10", "ndcg@10"):
        s4, sas = results["s4rec"][k], results["sasrec"][k]
        delta = 100.0 * (s4 - sas) / sas if sas else float("nan")
        logger.info("%-8s  S4Rec=%.4f  SASRec=%.4f  (%+.1f%%)", k, s4, sas, delta)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recommender", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to a YAML config")
    common.add_argument("--loss", default="ce", choices=["ce", "bce"])
    common.add_argument("--out", help="optional path to dump metrics JSON")

    sp = sub.add_parser("train", parents=[common], help="train a single model")
    sp.set_defaults(func=cmd_train)

    se = sub.add_parser("evaluate", parents=[common], help="evaluate a checkpoint")
    se.add_argument("--checkpoint", required=True)
    se.set_defaults(func=cmd_evaluate)

    sc = sub.add_parser("compare", parents=[common], help="S4Rec vs SASRec")
    sc.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
