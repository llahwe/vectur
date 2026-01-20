"""
CompGen: synthetic algorithmic datasets (plaintext {x_i, y_i} pairs).

Implements five heterogeneous task families referenced in `paper/paper.tex`:
- scan/reduce          (single pass)
- prefix sums          (linear auxiliary array)
- sort-then-scan       (sort + scan)
- nested-loop count    (quadratic pair count)
- DP table stripe      (quadratic DP with O(n) memory stripe)

The generator streams examples to disk (default ~1M) without storing them in RAM.

Output formats:
- jsonl (default): {"task": "...", "x": "...", "y": "...", "meta": {...}}
- tsv: "<x>\\t<y>\\n" (no meta)

Hugging Face:
- Optional upload/download helpers using `huggingface_hub` (if installed).
  No hard dependency: generation works without HF packages.

Example (generate ~1M examples per task, then upload to HF datasets):

    python3 -m datasets.compgen generate \\
      --out-dir datasets/compgen_runs \\
      --run-name compgen_v1 \\
      --dataset-size 1000000 \\
      --fmt jsonl \\
      --hf-user YOUR_HF_USERNAME \\
      --hf-repo YOUR_HF_USERNAME/compgen-compgen_v1 \\
      --hf-upload
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence


TaskName = Literal["scan_reduce", "prefix_sums", "sort_then_scan", "nested_loop_count", "dp_table_stripe"]
FormatName = Literal["jsonl", "tsv"]


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tsv_write(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y in rows:
            f.write(f"{x}\t{y}\n")


def _rand_int_list(rng: random.Random, *, n: int, lo: int, hi: int) -> list[int]:
    return [rng.randint(int(lo), int(hi)) for _ in range(int(n))]


def _format_int_list(xs: Sequence[int]) -> str:
    return ",".join(str(int(v)) for v in xs)


def _levenshtein_stripe(a: str, b: str) -> int:
    """
    Classic Levenshtein distance with O(len(b)) memory (DP stripe).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure b is the shorter dimension for memory (optional).
    if len(b) > len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = cur
    return int(prev[-1])


@dataclass(frozen=True)
class CompGenConfig:
    # Output
    out_dir: str = "datasets/compgen_runs"
    run_name: str | None = None
    fmt: FormatName = "jsonl"

    # Which tasks to generate
    tasks: tuple[TaskName, ...] = ("scan_reduce", "prefix_sums", "sort_then_scan", "nested_loop_count", "dp_table_stripe")

    # Size
    dataset_size: int = 1_000_000

    # Randomness
    seed: int = 1337

    # Problem sizing defaults
    min_n: int = 8
    # NOTE: For default ~1M examples, keep quadratic tasks tractable in Python.
    # You can raise this at generation time via CLI flags.
    max_n: int = 32
    int_lo: int = -50
    int_hi: int = 50
    alphabet: str = "abcd"
    min_str_len: int = 8
    max_str_len: int = 32

    # HF optional
    hf_user: str = "ethanhallphd"
    hf_repo: str | None = None  # if None, uses "{hf_user}/compgen-{run_name}"
    hf_private: bool = False


def gen_scan_reduce(rng: random.Random, *, n: int, lo: int, hi: int) -> tuple[str, str, dict[str, Any]]:
    ops = ("sum", "max", "min", "xor")
    op = rng.choice(ops)
    xs = _rand_int_list(rng, n=n, lo=lo, hi=hi)
    if op == "sum":
        y = sum(xs)
    elif op == "max":
        y = max(xs)
    elif op == "min":
        y = min(xs)
    else:
        acc = 0
        for v in xs:
            acc ^= int(v)
        y = acc
    x_txt = f"op={op}; arr={_format_int_list(xs)}"
    y_txt = str(int(y))
    meta = {"op": op, "n": int(n)}
    return x_txt, y_txt, meta


def gen_prefix_sums(rng: random.Random, *, n: int, lo: int, hi: int) -> tuple[str, str, dict[str, Any]]:
    xs = _rand_int_list(rng, n=n, lo=lo, hi=hi)
    out: list[int] = []
    s = 0
    for v in xs:
        s += int(v)
        out.append(int(s))
    x_txt = f"arr={_format_int_list(xs)}"
    y_txt = f"prefix={_format_int_list(out)}"
    meta = {"n": int(n)}
    return x_txt, y_txt, meta


def gen_sort_then_scan(rng: random.Random, *, n: int, lo: int, hi: int) -> tuple[str, str, dict[str, Any]]:
    """
    Sort-then-scan family: compute run-length encoding after sorting.
    Output is a compact string "v1:c1,v2:c2,...".
    """
    xs = _rand_int_list(rng, n=n, lo=lo, hi=hi)
    xs_sorted = sorted(xs)
    runs: list[tuple[int, int]] = []
    cur = xs_sorted[0]
    cnt = 1
    for v in xs_sorted[1:]:
        if v == cur:
            cnt += 1
        else:
            runs.append((int(cur), int(cnt)))
            cur = v
            cnt = 1
    runs.append((int(cur), int(cnt)))
    y_txt = "rle=" + ",".join(f"{v}:{c}" for v, c in runs)
    x_txt = f"arr={_format_int_list(xs)}"
    meta = {"n": int(n)}
    return x_txt, y_txt, meta


def gen_nested_loop_count(rng: random.Random, *, n: int, lo: int, hi: int) -> tuple[str, str, dict[str, Any]]:
    """
    Quadratic family: count pairs (i<j) such that arr[i] + arr[j] == target.
    """
    xs = _rand_int_list(rng, n=n, lo=lo, hi=hi)
    target = rng.randint(int(lo), int(hi))
    c = 0
    for i in range(len(xs)):
        xi = int(xs[i])
        for j in range(i + 1, len(xs)):
            if xi + int(xs[j]) == int(target):
                c += 1
    x_txt = f"target={int(target)}; arr={_format_int_list(xs)}"
    y_txt = str(int(c))
    meta = {"n": int(n), "target": int(target)}
    return x_txt, y_txt, meta


def gen_dp_table_stripe(rng: random.Random, *, a: str, min_len: int, max_len: int) -> tuple[str, str, dict[str, Any]]:
    """
    DP stripe family: Levenshtein distance between two random strings.
    """
    la = rng.randint(int(min_len), int(max_len))
    lb = rng.randint(int(min_len), int(max_len))
    s = "".join(rng.choice(a) for _ in range(la))
    t = "".join(rng.choice(a) for _ in range(lb))
    d = _levenshtein_stripe(s, t)
    x_txt = f"s={s}; t={t}"
    y_txt = f"dist={int(d)}"
    meta = {"len_s": int(la), "len_t": int(lb)}
    return x_txt, y_txt, meta


_TASK_GEN: dict[TaskName, Callable[[random.Random, CompGenConfig], tuple[str, str, dict[str, Any]]]] = {}


def _make_task_gen(task: TaskName) -> Callable[[random.Random, CompGenConfig], tuple[str, str, dict[str, Any]]]:
    def gen(rng: random.Random, cfg: CompGenConfig) -> tuple[str, str, dict[str, Any]]:
        n = rng.randint(int(cfg.min_n), int(cfg.max_n))
        if task == "scan_reduce":
            return gen_scan_reduce(rng, n=n, lo=cfg.int_lo, hi=cfg.int_hi)
        if task == "prefix_sums":
            return gen_prefix_sums(rng, n=n, lo=cfg.int_lo, hi=cfg.int_hi)
        if task == "sort_then_scan":
            return gen_sort_then_scan(rng, n=n, lo=cfg.int_lo, hi=cfg.int_hi)
        if task == "nested_loop_count":
            return gen_nested_loop_count(rng, n=n, lo=cfg.int_lo, hi=cfg.int_hi)
        if task == "dp_table_stripe":
            return gen_dp_table_stripe(rng, a=cfg.alphabet, min_len=cfg.min_str_len, max_len=cfg.max_str_len)
        raise ValueError(f"Unknown task: {task}")

    return gen


for _t in ("scan_reduce", "prefix_sums", "sort_then_scan", "nested_loop_count", "dp_table_stripe"):
    _TASK_GEN[_t] = _make_task_gen(_t)  # type: ignore[arg-type]


def generate_to_dir(cfg: CompGenConfig) -> Path:
    run_name = cfg.run_name or _now_tag()
    run_dir = Path(cfg.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "meta.json", dataclasses.asdict(cfg) | {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})

    for task in cfg.tasks:
        # Ensure each task file is reproducible independent of task order.
        task_seed = int(cfg.seed) ^ int(zlib.adler32(task.encode("utf-8")))
        rng = random.Random(task_seed)
        gen = _TASK_GEN[task]
        out_path = run_dir / f"{task}.{cfg.fmt}"

        if cfg.fmt == "jsonl":
            def _records() -> Iterator[dict[str, Any]]:
                for i in range(int(cfg.dataset_size)):
                    x, y, meta = gen(rng, cfg)
                    yield {"task": task, "x": x, "y": y, "meta": meta | {"i": i}}

            _jsonl_write(out_path, _records())
        else:
            def _rows() -> Iterator[tuple[str, str]]:
                for _i in range(int(cfg.dataset_size)):
                    x, y, _meta = gen(rng, cfg)
                    yield x, y

            _tsv_write(out_path, _rows())

    return run_dir


def hf_upload_folder(*, folder: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for upload. Install it with `pip install huggingface_hub`."
        ) from e

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", private=bool(private), exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=".",
    )


def hf_download_repo(*, repo_id: str, out_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for download. Install it with `pip install huggingface_hub`."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out_dir), local_dir_use_symlinks=False)
    return Path(local_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CompGen synthetic datasets (and optionally upload/download via HF).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate datasets into a new run directory under --out-dir.")
    g.add_argument("--out-dir", type=str, default="datasets/compgen_runs")
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--fmt", type=str, default="jsonl", choices=["jsonl", "tsv"])
    g.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list or 'all'. Options: scan_reduce,prefix_sums,sort_then_scan,nested_loop_count,dp_table_stripe",
    )
    g.add_argument("--dataset-size", type=int, default=1_000_000)
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--min-n", type=int, default=8)
    g.add_argument("--max-n", type=int, default=64)
    g.add_argument("--int-lo", type=int, default=-50)
    g.add_argument("--int-hi", type=int, default=50)
    g.add_argument("--alphabet", type=str, default="abcd")
    g.add_argument("--min-str-len", type=int, default=8)
    g.add_argument("--max-str-len", type=int, default=64)
    g.add_argument("--hf-user", type=str, default="ethanhallphd")
    g.add_argument("--hf-repo", type=str, default=None, help="Full repo id like 'user/name'. Defaults to '{hf_user}/compgen-{run_name}'.")
    g.add_argument("--hf-private", action="store_true")
    g.add_argument("--hf-upload", action="store_true", help="After generation, upload the run folder to HF datasets.")

    d = sub.add_parser("download", help="Download a dataset repo from HF to a local directory.")
    d.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id like 'ethanhallphd/compgen-...'.")
    d.add_argument("--out-dir", type=str, default="datasets/compgen_runs/downloads")

    u = sub.add_parser("upload", help="Upload an existing local run folder to HF datasets.")
    u.add_argument("--folder", type=str, required=True, help="Local run directory to upload.")
    u.add_argument("--hf-user", type=str, default="ethanhallphd")
    u.add_argument("--hf-repo", type=str, default=None, help="Full repo id like 'user/name'. Defaults to '{hf_user}/{folder_name}'.")
    u.add_argument("--hf-private", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cmd == "generate":
        if str(args.tasks).strip().lower() == "all":
            tasks: tuple[TaskName, ...] = ("scan_reduce", "prefix_sums", "sort_then_scan", "nested_loop_count", "dp_table_stripe")
        else:
            raw = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
            tasks = tuple(raw)  # type: ignore[assignment]
        run_name = args.run_name
        cfg = CompGenConfig(
            out_dir=str(args.out_dir),
            run_name=run_name,
            fmt=str(args.fmt),  # type: ignore[arg-type]
            tasks=tasks,
            dataset_size=int(args.dataset_size),
            seed=int(args.seed),
            min_n=int(args.min_n),
            max_n=int(args.max_n),
            int_lo=int(args.int_lo),
            int_hi=int(args.int_hi),
            alphabet=str(args.alphabet),
            min_str_len=int(args.min_str_len),
            max_str_len=int(args.max_str_len),
            hf_user=str(args.hf_user),
            hf_repo=(str(args.hf_repo) if args.hf_repo else None),
            hf_private=bool(args.hf_private),
        )
        run_dir = generate_to_dir(cfg)
        print(f"Wrote: {run_dir}")
        if bool(args.hf_upload):
            repo_id = cfg.hf_repo or f"{cfg.hf_user}/compgen-{run_dir.name}"
            hf_upload_folder(folder=run_dir, repo_id=repo_id, private=bool(cfg.hf_private))
            print(f"Uploaded to HF: {repo_id}")

    elif args.cmd == "download":
        out_dir = Path(str(args.out_dir))
        local = hf_download_repo(repo_id=str(args.repo_id), out_dir=out_dir / str(args.repo_id).replace("/", "__"))
        print(f"Downloaded to: {local}")

    elif args.cmd == "upload":
        folder = Path(str(args.folder))
        if not folder.exists():
            raise FileNotFoundError(folder)
        repo_id = str(args.hf_repo) if args.hf_repo else f"{str(args.hf_user)}/{folder.name}"
        hf_upload_folder(folder=folder, repo_id=repo_id, private=bool(args.hf_private))
        print(f"Uploaded to HF: {repo_id}")

    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

