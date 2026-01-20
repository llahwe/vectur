from __future__ import annotations

"""
Fine-tuning wrapper for ModularLlama.

This script exists to make "fine-tune stage" runs explicit, separate from pretraining.

Usage:
  - Either use --config (experiment config JSON), OR pass flags (not both).

Config format:
  Expects an "experiment config" JSON with:
    - finetune != "none"
    - finetune_train: TrainConfig-like object OR omitted (we will derive it)
    - train: TrainConfig-like object for pretraining (used only to find resume checkpoint if needed)

It ultimately calls into `train.py`'s training loop by importing the module.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Config must be a JSON object.")
    return obj


def _parse_args() -> argparse.Namespace:
    argv = list(sys.argv[1:])
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre.add_argument("-h", "--help", action="store_true")
    pre_ns, remaining = pre.parse_known_args(argv)

    if pre_ns.help:
        pass
    elif pre_ns.config is not None:
        if len(remaining) != 0:
            raise SystemExit("Error: `--config` cannot be combined with other flags.")
        return argparse.Namespace(config=str(pre_ns.config))

    p = argparse.ArgumentParser(description="Fine-tune an existing ModularLlama checkpoint.")
    p.add_argument("--resume", type=str, required=True, help="Path to base checkpoint (from pretraining).")
    p.add_argument("--run-name", type=str, required=True, help="Name for the fine-tune run.")
    p.add_argument("--train-config", type=str, required=True, help="Path to a TrainConfig JSON for fine-tuning.")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    # Import lazily so this file can be imported without torch installed.
    import train as train_mod  # type: ignore

    if getattr(args, "config", None):
        exp = _load_json(str(args.config))
        finetune = str(exp.get("finetune", "none"))
        if finetune == "none":
            raise ValueError("Config finetune='none' — nothing to fine-tune.")

        pretrain_train = exp.get("train") or {}
        if not isinstance(pretrain_train, dict):
            raise ValueError("Config 'train' must be an object.")

        finetune_train = exp.get("finetune_train")
        if finetune_train is None:
            # Derive a finetune TrainConfig by copying the pretrain TrainConfig
            # and overriding dataset fields from exp['finetune_dataset'] if present.
            finetune_train = dict(pretrain_train)
            ft_ds = exp.get("finetune_dataset")
            if isinstance(ft_ds, dict):
                for k, v in ft_ds.items():
                    finetune_train[k] = v

        if not isinstance(finetune_train, dict):
            raise ValueError("Config 'finetune_train' must be an object (or omitted).")

        # Determine resume checkpoint.
        resume = exp.get("resume_checkpoint")
        if resume is None:
            # Default: resume from the latest checkpoint of the pretrain run_name.
            run_name = pretrain_train.get("run_name")
            runs_dir = pretrain_train.get("runs_dir", "runs")
            if not run_name:
                raise ValueError("Config must provide train.run_name (to locate resume checkpoint) or resume_checkpoint.")
            resume = str(Path(str(runs_dir)) / str(run_name) / "checkpoints" / "latest.pt")

        finetune_train["resume"] = str(resume)
        if "run_name" not in finetune_train or not finetune_train["run_name"]:
            # Default finetune run name to the config stem if available.
            finetune_train["run_name"] = Path(str(args.config)).stem

        # Reuse train.py's JSON loader by writing an in-memory object:
        cfg = train_mod._load_train_config_json(str(args.config))  # type: ignore[attr-defined]
        # The above would load exp['train'], not finetune; so instead load directly:
        cfg = train_mod._load_train_config_json(str(_write_tmp_train_config(finetune_train)))  # type: ignore[attr-defined]
        train_mod.main_with_cfg(cfg)  # type: ignore[attr-defined]

    else:
        # Flag mode: load TrainConfig JSON, inject resume + run_name, and train.
        train_cfg_path = Path(str(args.train_config))
        raw = json.loads(train_cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("--train-config must be a JSON object.")
        raw["resume"] = str(args.resume)
        raw["run_name"] = str(args.run_name)
        cfg_path = _write_tmp_train_config(raw)
        cfg = train_mod._load_train_config_json(str(cfg_path))  # type: ignore[attr-defined]
        train_mod.main_with_cfg(cfg)  # type: ignore[attr-defined]


def _write_tmp_train_config(obj: dict[str, Any]) -> Path:
    """
    Write a temporary TrainConfig JSON into the same directory as this script.
    """
    out_dir = Path(__file__).resolve().parent / ".tmp_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "finetune_tmp.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()

