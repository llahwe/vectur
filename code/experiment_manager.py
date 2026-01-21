from __future__ import annotations

import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Iterable, Literal

import socket


Status = Literal[
    "not_started",
    "in_progress",
    "completed",
    "failed",
    "interrupted",
]

# CompGen HF datasets repo id (see `vectur/datasets/compgen.py`).
# Note: this dataset is independent of model size, so do not suffix with _s/_m/_l.
COMPGEN_DATASET_ID = "ethanhallphd/compgen-compgen_v1"


def _utc_ts() -> float:
    return float(time.time())


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(obj)}")
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_done(status: Status) -> bool:
    return status == "completed"


def _is_blocking(status: Status) -> bool:
    return status in ("failed", "interrupted")


@dataclass
class WorkNode:
    id: str
    action: Literal["train", "finetune", "eval", "paper", "noop"]
    depends_on: list[str]
    status: Status = "not_started"
    created_utc: float = dataclasses.field(default_factory=_utc_ts)
    updated_utc: float = dataclasses.field(default_factory=_utc_ts)
    attempts: int = 0
    last_exit_code: int | None = None
    last_error: str | None = None
    # Free-form spec interpreted by the manager (this is the "source of truth" per node).
    spec: dict[str, Any] = dataclasses.field(default_factory=dict)
    log_path: str | None = None  # optional override

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkNode":
        if not isinstance(d, dict):
            raise ValueError(f"WorkNode must be built from an object, got {type(d)}")

        # Backward-compatible load: tolerate legacy keys from older graphs.
        if "action" not in d:
            legacy_kind = str(d.get("kind", "noop"))
            action: str = legacy_kind
            if action not in ("train", "finetune", "eval", "paper"):
                action = "noop"
            status_map = {
                "pending": "not_started",
                "in_progress": "in_progress",
                "completed": "completed",
                "failed": "failed",
                "skipped": "completed",
                "manual_pending": "not_started",
                "manual_completed": "completed",
                "manual_failed": "failed",
            }
            legacy_status = str(d.get("status", "not_started"))
            status = status_map.get(legacy_status, "not_started")
            return cls(
                id=str(d.get("id") or ""),
                action=action,  # type: ignore[arg-type]
                depends_on=list(d.get("depends_on") or []),
                status=status,  # type: ignore[arg-type]
                created_utc=float(d.get("created_utc", _utc_ts())),
                updated_utc=float(d.get("updated_utc", _utc_ts())),
                attempts=int(d.get("attempts", 0)),
                last_exit_code=(int(d["last_exit_code"]) if d.get("last_exit_code") is not None else None),
                last_error=(str(d["last_error"]) if d.get("last_error") is not None else None),
                spec={},
                log_path=(str(d["log_path"]) if d.get("log_path") else None),
            )

        # New schema: drop unknown keys defensively.
        fields = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in d.items() if k in fields}
        return cls(**clean)  # type: ignore[arg-type]


@dataclass
class ExperimentManager:
    """
    Maintains a persisted DAG of work nodes derived from `work_graph.json`.

    The DAG is persisted as JSON so we can:
    - skip completed stages on rerun
    - manually override status by editing the JSON (e.g. interrupted but mark completed)
    """

    def __init__(
        self,
        *,
        graph_path: Path,
        state_dir: Path,
        python: str = "python3",
        # Remote state + locks via rclone (e.g. Google Drive).
        rclone_remote: str | None = None,
        rclone_root: str | None = None,
        owner_id: str | None = None,
        # IMPORTANT: stage locks are not refreshed while a stage is running, so TTL must be
        # longer than your longest expected stage runtime (e.g. multi-day training).
        lock_ttl_seconds: int = 4 * 24 * 3600,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.state_dir = Path(state_dir)
        self.python = str(python)
        self.nodes: dict[str, WorkNode] = {}

        # Remote state configuration:
        # - If rclone_remote/root are provided (or env vars), we will:
        #   - pull/push the work graph JSON through rclone
        #   - use lock files on the remote as mutexes
        env_remote = os.environ.get("RCLONE_REMOTE")
        env_root = os.environ.get("RCLONE_ROOT")
        self.rclone_remote = (rclone_remote or env_remote or "").strip() or None
        self.rclone_root = (rclone_root or env_root or "").strip().strip("/") or None
        self.remote_enabled = bool(self.rclone_remote and self.rclone_root)

        host = socket.gethostname()
        pid = os.getpid()
        self.owner_id = str(owner_id or os.environ.get("OWNER_ID") or f"{host}:{pid}:{uuid.uuid4().hex[:8]}")
        self.lock_ttl_seconds = int(lock_ttl_seconds)

        # Where the authoritative graph JSON and lockfiles live on the remote.
        # Prefer to preserve the path relative to code_dir (so we can store e.g. "experiments/work_graph.json"
        # on the remote). Fall back to basename for paths outside the code directory.
        #
        # Backward compat: older versions stored only the basename at the remote root.
        try:
            self._remote_graph_rel = str(self.graph_path.resolve().relative_to(self.code_dir().resolve())).replace("\\", "/")
        except Exception:
            self._remote_graph_rel = self.graph_path.name
        self._remote_graph_rel_legacy = self.graph_path.name
        self._remote_locks_rel = "locks"
        self._remote_stage_locks_rel = "locks/stages"
        self._remote_graph_lock_rel = "locks/graph.lock.json"

    def repo_root(self) -> Path:
        # This file lives in vectur/code/; repo root is vectur/
        return Path(__file__).resolve().parents[1]

    def code_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _safe_name(self, s: str) -> str:
        return s.replace("::", "__").replace("/", "_").replace(" ", "_")

    def _remote_path(self, rel: str) -> str:
        if not self.remote_enabled:
            raise RuntimeError("Remote is not enabled (missing rclone_remote/rclone_root).")
        root = str(self.rclone_root or "").strip("/").rstrip("/")
        rel2 = str(rel).lstrip("/")
        # rclone expects "remote:dir/file"
        return f"{self.rclone_remote}{root}/{rel2}"

    def _rclone(self, args: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["rclone", *args]
        if capture:
            return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)
        return subprocess.run(cmd, text=True, check=check)

    def remote_pull_graph(self) -> None:
        if not self.remote_enabled:
            return
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        # Best-effort: prefer new path; fall back to legacy basename path.
        remote_new = self._remote_path(self._remote_graph_rel)
        print(f"[ExperimentManager] Pulling graph from {remote_new} ...")
        cp = self._rclone(["copyto", remote_new, str(self.graph_path)], capture=True, check=False)
        if cp.returncode == 0:
            return
        print(f"[ExperimentManager] Pull failed (new path): {cp.stderr.strip() or 'unknown error'}")
        
        remote_legacy = self._remote_path(self._remote_graph_rel_legacy)
        print(f"[ExperimentManager] Trying legacy path {remote_legacy} ...")
        cp2 = self._rclone(["copyto", remote_legacy, str(self.graph_path)], capture=True, check=False)
        if cp2.returncode != 0:
            print(f"[ExperimentManager] Pull failed (legacy path): {cp2.stderr.strip() or 'unknown error'}")
        _ = cp2  # ignore errors ("not found" on first run)

    def remote_push_graph(self) -> None:
        if not self.remote_enabled:
            return
        if not self.graph_path.exists():
            return
        # Ensure directories exist.
        self._rclone(["mkdir", self._remote_path(self._remote_locks_rel)], check=False)
        self._rclone(["mkdir", self._remote_path(self._remote_stage_locks_rel)], check=False)
        # Ensure parent directory exists for the graph (best-effort).
        try:
            rel = str(self._remote_graph_rel).replace("\\", "/")
            parent_rel = rel.rsplit("/", 1)[0] if "/" in rel else ""
            if parent_rel:
                self._rclone(["mkdir", self._remote_path(parent_rel)], check=False)
        except Exception:
            pass
        remote = self._remote_path(self._remote_graph_rel)
        print(f"[ExperimentManager] Pushing graph to {remote} ...")
        self._rclone(["copyto", str(self.graph_path), remote], check=True)
        # Also write legacy location for older workers (best-effort).
        remote_legacy = self._remote_path(self._remote_graph_rel_legacy)
        if remote_legacy != remote:
            self._rclone(["copyto", str(self.graph_path), remote_legacy], check=False)

    def _remote_lock_path(self, rel: str) -> str:
        return self._remote_path(rel)

    def _read_remote_text(self, remote_path: str) -> str | None:
        if not self.remote_enabled:
            return None
        cp = self._rclone(["cat", remote_path], capture=True, check=False)
        if cp.returncode != 0:
            return None
        return cp.stdout

    def _remote_exists(self, remote_path: str) -> bool:
        if not self.remote_enabled:
            return False
        cp = self._rclone(["lsf", remote_path], capture=True, check=False)
        return cp.returncode == 0

    def _lock_payload(self) -> dict[str, Any]:
        now = _utc_ts()
        return {
            "owner_id": self.owner_id,
            "host": socket.gethostname(),
            "pid": int(os.getpid()),
            "acquired_utc": now,
            "expires_utc": now + float(self.lock_ttl_seconds),
        }

    def _try_acquire_remote_lock(self, *, lock_rel: str) -> bool:
        """
        Acquire a remote lock by writing a JSON lockfile.

        Protocol:
        - If lock file absent: create with `rclone copyto --ignore-existing`, then verify.
        - If present and expired: overwrite, then verify.
        - Otherwise: fail.
        """
        if not self.remote_enabled:
            return True

        remote = self._remote_lock_path(lock_rel)
        # Ensure parent directory exists on remote (best-effort).
        try:
            parent_rel = str(lock_rel).replace("\\", "/")
            parent_rel = parent_rel.rsplit("/", 1)[0] if "/" in parent_rel else ""
            if parent_rel:
                self._rclone(["mkdir", self._remote_path(parent_rel)], check=False)
        except Exception:
            pass
        local_dir = self.state_dir / "remote_locks"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"{self._safe_name(lock_rel)}.json"
        _write_json(local_path, self._lock_payload())

        # 1) Try create-if-missing.
        cp = self._rclone(["copyto", str(local_path), remote, "--ignore-existing"], capture=True, check=False)

        # 2) Verify ownership if we might have created it.
        txt = self._read_remote_text(remote)
        if txt:
            try:
                obj = json.loads(txt)
            except Exception:
                obj = None
            if isinstance(obj, dict) and obj.get("owner_id") == self.owner_id:
                return True

        # 3) If it exists but is expired, attempt takeover.
        if txt:
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict) and float(obj.get("expires_utc", 0.0)) < _utc_ts():
                    self._rclone(["copyto", str(local_path), remote], capture=True, check=False)
                    txt2 = self._read_remote_text(remote)
                    if txt2:
                        obj2 = json.loads(txt2)
                        if isinstance(obj2, dict) and obj2.get("owner_id") == self.owner_id:
                            return True
            except Exception:
                pass

        # Could not acquire.
        _ = cp  # keep for debugging if needed
        return False

    def _release_remote_lock(self, *, lock_rel: str) -> None:
        if not self.remote_enabled:
            return
        remote = self._remote_lock_path(lock_rel)
        txt = self._read_remote_text(remote)
        if not txt:
            return
        try:
            obj = json.loads(txt)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        if obj.get("owner_id") != self.owner_id:
            # Do not delete another worker's lock.
            return
        self._rclone(["deletefile", remote], check=False)

    @contextmanager
    def graph_mutex(self) -> Iterable[None]:
        """
        A global mutex that serializes work-graph updates across multiple workers.
        """
        if self.remote_enabled:
            ok = self._try_acquire_remote_lock(lock_rel=self._remote_graph_lock_rel)
            if not ok:
                raise RuntimeError("Could not acquire global graph lock (remote).")
        try:
            yield
        finally:
            if self.remote_enabled:
                self._release_remote_lock(lock_rel=self._remote_graph_lock_rel)

    def generate_initial_work_items(self, *, overwrite: bool = False) -> None:
        """
        Generate an initial, portable `work_graph.json` with shared atomic work items.

        This does NOT create per-stage config files; the graph is the sole source of truth.
        """
        if self.graph_path.exists() and not overwrite:
            raise FileExistsError(f"{self.graph_path} already exists (pass overwrite=True to replace).")

        nodes: dict[str, WorkNode] = {}

        blocks = ["attention", "lstm", "ntm", "vectur", "vecstur"]
        # Default to small models for sanity / cost control.
        # TrainConfig.model_size expects one of {"s","m","l"}.
        size = "s"

        def train_node_id(block: str, trainset: str) -> str:
            return f"train::{block}::{trainset}::{size}"

        def dry_train_node_id(block: str, trainset: str) -> str:
            return f"dry_run_train::{block}::{trainset}::{size}"

        def finetune_node_id(block: str, base: str, ft: str) -> str:
            return f"finetune::{block}::{base}_to_{ft}::{size}"

        def dry_finetune_node_id(block: str, base: str, ft: str) -> str:
            return f"dry_run_finetune::{block}::{base}_to_{ft}::{size}"

        def eval_node_id(tag: str, block: str, ckpt_run: str) -> str:
            return f"eval::{tag}::{block}::{ckpt_run}"

        def dry_eval_node_id(tag: str, block: str, ckpt_run: str) -> str:
            return f"dry_run_eval::{tag}::{block}::{ckpt_run}"

        # Shared training work items.
        for block in blocks:
            # FineWeb pretrain (protocol 1)
            run_name_fw = f"{block}_fineweb_{size}"
            tid = train_node_id(block, "fineweb")
            did = dry_train_node_id(block, "fineweb")
            train_cfg_fw = {
                "run_name": run_name_fw,
                "runs_dir": "runs",
                "model_size": size,
                "block": block,
                "n_layers": 0,
                "data_source": "hf_buffer",
                "dataset_name": "HuggingFaceFW/fineweb-edu",
                "dataset_config": "sample-10BT",
                "dataset_split": "train",
                "dataset_text_field": "text",
                "tokenizer_name": "gpt2",
            }
            nodes[did] = WorkNode(
                id=did,
                action="train",
                depends_on=[],
                status="not_started",
                spec={"train_config": {**train_cfg_fw, "dry_run": True, "run_name": f"{run_name_fw}__dryrun"}},
            )
            nodes[tid] = WorkNode(
                id=tid,
                action="train",
                depends_on=[did],
                status="not_started",
                spec={"train_config": train_cfg_fw},
            )

            # CLRS train (protocol 2/3)
            run_name_clrs = f"{block}_clrs_{size}"
            tid = train_node_id(block, "clrs")
            did = dry_train_node_id(block, "clrs")
            train_cfg_clrs = {
                "run_name": run_name_clrs,
                "runs_dir": "runs",
                "model_size": size,
                "block": block,
                "n_layers": 0,
                "data_source": "hf_buffer",
                "dataset_name": "tomg-group-umd/CLRS-Text-train",
                "dataset_config": None,
                "dataset_split": "train",
                "dataset_text_template": "algo={algo_name}\n q: {question}\n a: {answer}\n",
                "tokenizer_name": "gpt2",
            }
            nodes[did] = WorkNode(
                id=did,
                action="train",
                depends_on=[],
                status="not_started",
                spec={"train_config": {**train_cfg_clrs, "dry_run": True, "run_name": f"{run_name_clrs}__dryrun"}},
            )
            nodes[tid] = WorkNode(
                id=tid,
                action="train",
                depends_on=[did],
                status="not_started",
                spec={"train_config": train_cfg_clrs},
            )

            # CompGen train (protocol 2b)
            run_name_compgen = f"{block}_compgen_{size}"
            tid = train_node_id(block, "compgen")
            did = dry_train_node_id(block, "compgen")
            train_cfg_cg = {
                "run_name": run_name_compgen,
                "runs_dir": "runs",
                "model_size": size,
                "block": block,
                "n_layers": 0,
                "data_source": "hf_buffer",
                "dataset_name": COMPGEN_DATASET_ID,
                "dataset_config": None,
                "dataset_split": "train",
                "dataset_text_template": "task={task}\n{x}\n{y}\n",
                "tokenizer_name": "gpt2",
            }
            nodes[did] = WorkNode(
                id=did,
                action="train",
                depends_on=[],
                status="not_started",
                spec={"train_config": {**train_cfg_cg, "dry_run": True, "run_name": f"{run_name_compgen}__dryrun"}},
            )
            nodes[tid] = WorkNode(
                id=tid,
                action="train",
                depends_on=[did],
                status="not_started",
                spec={"train_config": train_cfg_cg},
            )

            # Fine-tune CLRS → CompGen (protocol 2a fine-tune)
            base_train = train_node_id(block, "clrs")
            ft_run = f"{block}_clrs_to_compgenft_{size}"
            dry_ft_id = dry_finetune_node_id(block, "clrs", "compgen")
            ft_id = finetune_node_id(block, "clrs", "compgen")
            ft_exp_cfg = {
                "train": {"run_name": run_name_clrs, "runs_dir": "runs"},
                "finetune": "supervised",
                "resume_checkpoint": f"runs/{run_name_clrs}/checkpoints/latest.pt",
                "finetune_train": {
                    "run_name": ft_run,
                    "runs_dir": "runs",
                    "model_size": size,
                    "block": block,
                    "n_layers": 0,
                    "data_source": "hf_buffer",
                    "dataset_name": "ethanhallphd/compgen-compgen_v1_m",
                    "dataset_config": None,
                    "dataset_split": "train",
                    "dataset_text_template": "task={task}\n{x}\n{y}\n",
                    "tokenizer_name": "gpt2",
                },
            }
            nodes[dry_ft_id] = WorkNode(
                id=dry_ft_id,
                action="finetune",
                depends_on=[base_train],
                status="not_started",
                spec={"experiment_config": {**ft_exp_cfg, "finetune_train": {**ft_exp_cfg["finetune_train"], "dry_run": True, "run_name": f"{ft_run}__dryrun"}}},
            )
            nodes[ft_id] = WorkNode(
                id=ft_id,
                action="finetune",
                depends_on=[dry_ft_id],
                status="not_started",
                spec={"experiment_config": ft_exp_cfg},
            )

            # Eval: FineWeb downstream (protocol 1) on HF datasets
            # We use perplexity-style eval via token buffers (not accuracy metrics).
            fw_ckpt = f"runs/{run_name_fw}/checkpoints/latest.pt"
            downstream = [
                (
                    "gsm8k_test",
                    {
                        "name": "gsm8k_test",
                        "kind": "ppl",
                        "data_source": "hf_buffer",
                        "dataset_name": "openai/gsm8k",
                        "dataset_config": "main",
                        "dataset_split": "test",
                        "dataset_text_template": "Q: {question}\nA: {answer}\n",
                        "dataset_text_field": "question",
                        "tokenizer_name": "gpt2",
                    },
                ),
                (
                    "arc_challenge_test",
                    {
                        "name": "arc_challenge_test",
                        "kind": "ppl",
                        "data_source": "hf_buffer",
                        "dataset_name": "allenai/ai2_arc",
                        "dataset_config": "ARC-Challenge",
                        "dataset_split": "test",
                        "dataset_text_template": "Q: {question}\nChoices: {choices}\nA: {answerKey}\n",
                        "dataset_text_field": "question",
                        "tokenizer_name": "gpt2",
                    },
                ),
                (
                    "hellaswag_val",
                    {
                        "name": "hellaswag_val",
                        "kind": "ppl",
                        "data_source": "hf_buffer",
                        "dataset_name": "Rowan/hellaswag",
                        "dataset_config": None,
                        "dataset_split": "validation",
                        "dataset_text_template": "{ctx}\n{endings}\nlabel={label}\n",
                        "dataset_text_field": "ctx",
                        "tokenizer_name": "gpt2",
                    },
                ),
                (
                    "wikitext103_test",
                    {
                        "name": "wikitext103_test",
                        "kind": "ppl",
                        "data_source": "hf_buffer",
                        "dataset_name": "Salesforce/wikitext",
                        "dataset_config": "wikitext-103-v1",
                        "dataset_split": "test",
                        "dataset_text_field": "text",
                        "tokenizer_name": "gpt2",
                    },
                ),
            ]
            for tag, test in downstream:
                deid = dry_eval_node_id(tag, block, run_name_fw)
                eid = eval_node_id(tag, block, run_name_fw)
                nodes[deid] = WorkNode(
                    id=deid,
                    action="eval",
                    depends_on=[train_node_id(block, "fineweb")],
                    status="not_started",
                    spec={
                        "eval_config": {
                            "checkpoint": fw_ckpt,
                            "eval": {"batch_size": 2, "seq_len": 256, "max_batches": 2},
                            "tests": [test],
                        }
                    },
                )
                nodes[eid] = WorkNode(
                    id=eid,
                    action="eval",
                    depends_on=[deid],
                    status="not_started",
                    spec={
                        "eval_config": {
                            "checkpoint": fw_ckpt,
                            "tests": [test],
                        }
                    },
                )

            # Eval: CLRS held-out (protocol 3)
            clrs_ckpt = f"runs/{run_name_clrs}/checkpoints/latest.pt"
            clrs_test = {
                "name": "clrs_text_test",
                "kind": "ppl",
                "data_source": "hf_buffer",
                "dataset_name": "tomg-group-umd/CLRS-Text-test",
                "dataset_config": None,
                "dataset_split": "test",
                "dataset_text_template": "algo={algo_name}\n q: {question}\n a: {answer}\n",
                "tokenizer_name": "gpt2",
            }
            deid = dry_eval_node_id("clrs_text_test", block, run_name_clrs)
            eid = eval_node_id("clrs_text_test", block, run_name_clrs)
            nodes[deid] = WorkNode(
                id=deid,
                action="eval",
                depends_on=[train_node_id(block, "clrs")],
                status="not_started",
                spec={"eval_config": {"checkpoint": clrs_ckpt, "eval": {"batch_size": 2, "seq_len": 256, "max_batches": 2}, "tests": [clrs_test]}},
            )
            nodes[eid] = WorkNode(
                id=eid,
                action="eval",
                depends_on=[deid],
                status="not_started",
                spec={"eval_config": {"checkpoint": clrs_ckpt, "tests": [clrs_test]}},
            )

            # Eval: CompGen held-out (protocol 2b)
            cg_ckpt = f"runs/{run_name_compgen}/checkpoints/latest.pt"
            cg_test = {
                "name": "compgen_test",
                "kind": "ppl",
                "data_source": "hf_buffer",
                "dataset_name": COMPGEN_DATASET_ID,
                "dataset_config": None,
                "dataset_split": "test",
                "dataset_text_template": "task={task}\n{x}\n{y}\n",
                "tokenizer_name": "gpt2",
            }
            deid = dry_eval_node_id("compgen_test", block, run_name_compgen)
            eid = eval_node_id("compgen_test", block, run_name_compgen)
            nodes[deid] = WorkNode(
                id=deid,
                action="eval",
                depends_on=[train_node_id(block, "compgen")],
                status="not_started",
                spec={"eval_config": {"checkpoint": cg_ckpt, "eval": {"batch_size": 2, "seq_len": 256, "max_batches": 2}, "tests": [cg_test]}},
            )
            nodes[eid] = WorkNode(
                id=eid,
                action="eval",
                depends_on=[deid],
                status="not_started",
                spec={"eval_config": {"checkpoint": cg_ckpt, "tests": [cg_test]}},
            )

            # Eval: CLRS→CompGen zero-shot (protocol 2a zero-shot)
            cg_zero_test = {**cg_test, "name": "compgen_test_zero_shot"}
            deid = dry_eval_node_id("compgen_test_zero_shot", block, run_name_clrs)
            eid = eval_node_id("compgen_test_zero_shot", block, run_name_clrs)
            nodes[deid] = WorkNode(
                id=deid,
                action="eval",
                depends_on=[train_node_id(block, "clrs")],
                status="not_started",
                spec={"eval_config": {"checkpoint": clrs_ckpt, "eval": {"batch_size": 2, "seq_len": 256, "max_batches": 2}, "tests": [cg_zero_test]}},
            )
            nodes[eid] = WorkNode(
                id=eid,
                action="eval",
                depends_on=[deid],
                status="not_started",
                spec={"eval_config": {"checkpoint": clrs_ckpt, "tests": [cg_zero_test]}},
            )

            # Eval: CLRS→CompGen after fine-tune (protocol 2a fine-tune)
            ft_ckpt = f"runs/{ft_run}/checkpoints/latest.pt"
            cg_ft_test = {**cg_test, "name": "compgen_test_finetuned"}
            deid = dry_eval_node_id("compgen_test_finetuned", block, ft_run)
            eid = eval_node_id("compgen_test_finetuned", block, ft_run)
            nodes[deid] = WorkNode(
                id=deid,
                action="eval",
                depends_on=[finetune_node_id(block, "clrs", "compgen")],
                status="not_started",
                spec={"eval_config": {"checkpoint": ft_ckpt, "eval": {"batch_size": 2, "seq_len": 256, "max_batches": 2}, "tests": [cg_ft_test]}},
            )
            nodes[eid] = WorkNode(
                id=eid,
                action="eval",
                depends_on=[deid],
                status="not_started",
                spec={"eval_config": {"checkpoint": ft_ckpt, "tests": [cg_ft_test]}},
            )

        payload = {
            "version": 2,
            "updated_utc": _utc_ts(),
            "nodes": {k: v.to_dict() for k, v in nodes.items()},
            "remote": {
                "enabled": bool(self.remote_enabled),
                "rclone_remote": self.rclone_remote,
                "rclone_root": self.rclone_root,
            },
        }
        _write_json(self.graph_path, payload)
        print(f"[ExperimentManager] Generated initial graph at {self.graph_path}")
        self.remote_push_graph()

    def _tmp_cfg_path(self, node_id: str) -> Path:
        safe = self._safe_name(node_id)
        p = self.state_dir / "tmp_configs" / f"{safe}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _node_log_path(self, node_id: str, override: str | None = None) -> Path:
        if override:
            return Path(override)
        safe = self._safe_name(node_id)
        return self.state_dir / "logs" / f"{safe}.log"

    def discover_configs(self) -> list[Path]:
        raise NotImplementedError("Configs discovery is deprecated; use work_graph.json as source of truth.")

    def load_experiments(self) -> None:
        raise NotImplementedError("Experiments discovery is deprecated; use work_graph.json as source of truth.")

    def _is_experiment_config(self, cfg: dict[str, Any]) -> bool:
        # Heuristic: experiment config wraps train config under "train" and may include tests/finetune.
        return isinstance(cfg.get("train"), dict) or isinstance(cfg.get("tests"), list) or "finetune" in cfg

    def _get_train_cfg(self, cfg: dict[str, Any]) -> dict[str, Any]:
        if self._is_experiment_config(cfg):
            train = cfg.get("train") or {}
            if not isinstance(train, dict):
                raise ValueError("experiment config field 'train' must be an object")
            return train
        return cfg

    def _stage_config_path(self, node_id: str) -> Path:
        safe = node_id.replace("::", "__").replace("/", "_")
        return self.state_dir / "stage_configs" / f"{safe}.json"

    def _log_path(self, node_id: str) -> Path:
        safe = node_id.replace("::", "__").replace("/", "_")
        return self.state_dir / "logs" / f"{safe}.log"

    def _make_stage_experiment_config(self, *, base: dict[str, Any], train_override: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._is_experiment_config(base):
            out = dict(base)
            if train_override is not None:
                out["train"] = train_override
            return out
        # Wrap TrainConfig into experiment config.
        return {"train": (train_override if train_override is not None else dict(base))}

    def _command_train(self, config_path: Path) -> list[str]:
        python_parts = self.python.split() if " " in self.python else [self.python]
        return [*python_parts, str(Path(__file__).resolve().parent / "train.py"), "--config", str(config_path)]

    def _command_finetune(self, config_path: Path) -> list[str]:
        python_parts = self.python.split() if " " in self.python else [self.python]
        return [*python_parts, str(Path(__file__).resolve().parent / "fine_tune.py"), "--config", str(config_path)]

    def _command_eval(self, config_path: Path) -> list[str]:
        python_parts = self.python.split() if " " in self.python else [self.python]
        return [*python_parts, str(Path(__file__).resolve().parent / "eval.py"), "--config", str(config_path)]

    def build_or_update_graph(self) -> None:
        raise NotImplementedError("Graph building from configs is deprecated; edit or regenerate work_graph.json instead.")

    def _persist(self) -> None:
        payload = {
            "version": 2,
            "updated_utc": _utc_ts(),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "remote": {
                "enabled": bool(self.remote_enabled),
                "rclone_remote": self.rclone_remote,
                "rclone_root": self.rclone_root,
            },
        }
        _write_json(self.graph_path, payload)
        self.remote_push_graph()

    def _load_graph_file(self) -> "ExperimentManager":
        raw = _read_json(self.graph_path)
        nodes_raw = raw.get("nodes", {})
        if not isinstance(nodes_raw, dict):
            raise ValueError("graph file nodes must be an object")
        self.nodes = {k: WorkNode.from_dict(v) for k, v in nodes_raw.items()}
        return self

    def _validate_graph(self) -> None:
        """
        Defensive validation to avoid accidentally running the wrong thing.
        - Require node.id to match its key (when provided).
        - Require all dependencies to exist.
        - Require action-specific spec payloads for runnable actions.
        """
        for k, n in self.nodes.items():
            if n.id and str(n.id) != str(k):
                raise ValueError(f"Graph node key/id mismatch: key={k!r} node.id={n.id!r}")
            for dep in n.depends_on:
                if dep not in self.nodes:
                    raise KeyError(f"Node {k!r} depends on missing node {dep!r}")
            if n.action == "train":
                if not isinstance(n.spec.get("train_config"), dict):
                    raise ValueError(f"{k!r}: train node missing spec.train_config object")
            elif n.action == "finetune":
                if not isinstance(n.spec.get("experiment_config"), dict):
                    raise ValueError(f"{k!r}: finetune node missing spec.experiment_config object")
            elif n.action == "eval":
                if not isinstance(n.spec.get("eval_config"), dict):
                    raise ValueError(f"{k!r}: eval node missing spec.eval_config object")

    def load_graph(self) -> None:
        # Best-effort: prefer remote as source of truth when enabled.
        self.remote_pull_graph()
        if self.graph_path.exists():
            self._load_graph_file()
            self._validate_graph()

    def refresh(self) -> None:
        # No-op: graph is authoritative and edited manually or via generate_initial_work_items().
        self.load_graph()

    def set_status(self, node_id: str, status: Status, note: str | None = None) -> None:
        with self.graph_mutex():
            self.remote_pull_graph()
            if self.graph_path.exists():
                self._load_graph_file()
            if node_id not in self.nodes:
                raise KeyError(node_id)
            n = self.nodes[node_id]
            n.status = status
            n.updated_utc = _utc_ts()
            if note:
                n.last_error = str(note)
            self._persist()

    def _stage_lock_rel(self, node_id: str) -> str:
        safe = self._safe_name(node_id)
        return f"{self._remote_stage_locks_rel}/{safe}.lock.json"

    def runnable_nodes(self) -> list[WorkNode]:
        out: list[WorkNode] = []
        for n in self.nodes.values():
            if n.status != "not_started":
                continue
            # Do not auto-run non-executable nodes.
            if n.action in ("paper", "noop"):
                continue
            # Dependencies are validated on load; if something is missing anyway, do not run.
            if any(d not in self.nodes for d in n.depends_on):
                continue
            deps = [self.nodes[d] for d in n.depends_on]
            if any(_is_blocking(d.status) for d in deps):
                continue
            if all(_is_done(d.status) for d in deps):
                out.append(n)
        return sorted(out, key=lambda x: x.id)

    def run_node(self, node_id: str) -> tuple[bool, str]:
        stage_lock_rel = self._stage_lock_rel(node_id)

        # Acquire stage lock + mark in progress under a global graph mutex.
        with self.graph_mutex():
            self.remote_pull_graph()
            if self.graph_path.exists():
                self._load_graph_file()
            if node_id not in self.nodes:
                raise KeyError(node_id)
            n = self.nodes[node_id]
            if n.status != "not_started":
                return False, f"Not runnable (status={n.status})"
            if n.action in ("paper", "noop"):
                return False, f"Not runnable (action={n.action})"

            # Stage mutex prevents duplicate work across parallel workers.
            if not self._try_acquire_remote_lock(lock_rel=stage_lock_rel):
                return False, "Locked (another worker owns this stage)"

            n.status = "in_progress"
            n.updated_utc = _utc_ts()
            n.attempts += 1
            self._persist()

        # Build command from node spec (graph is the sole source of truth).
        # Split python command if it contains spaces (e.g., "uv run python3" -> ["uv", "run", "python3"])
        python_parts = self.python.split() if " " in self.python else [self.python]
        cmd: list[str]
        cwd = str(self.code_dir())
        if n.action == "train":
            train_cfg = n.spec.get("train_config")
            if not isinstance(train_cfg, dict):
                raise ValueError(f"{node_id}: spec.train_config must be an object")
            tmp = self._tmp_cfg_path(node_id)
            _write_json(tmp, train_cfg)
            cmd = [*python_parts, str(self.code_dir() / "train.py"), "--config", str(tmp)]
        elif n.action == "finetune":
            exp_cfg = n.spec.get("experiment_config")
            if not isinstance(exp_cfg, dict):
                raise ValueError(f"{node_id}: spec.experiment_config must be an object")
            tmp = self._tmp_cfg_path(node_id)
            _write_json(tmp, exp_cfg)
            cmd = [*python_parts, str(self.code_dir() / "fine_tune.py"), "--config", str(tmp)]
        elif n.action == "eval":
            eval_cfg = n.spec.get("eval_config")
            if not isinstance(eval_cfg, dict):
                raise ValueError(f"{node_id}: spec.eval_config must be an object")
            tmp = self._tmp_cfg_path(node_id)
            _write_json(tmp, eval_cfg)
            cmd = [*python_parts, str(self.code_dir() / "eval.py"), "--config", str(tmp)]
        else:
            raise ValueError(f"{node_id}: unknown action={n.action}")

        # Execute outside the graph mutex so other stages can be claimed.
        log_path = self._node_log_path(node_id, override=n.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        ok = False
        msg = "Unknown"
        exit_code: int | None = None
        err_text: str | None = None
        try:
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"\n=== {time.ctime()} RUN {n.attempts} ===\n")
                lf.write(f"owner_id: {self.owner_id}\n")
                lf.write(f"cwd: {cwd}\n")
                lf.write(f"cmd: {' '.join(cmd)}\n\n")
                lf.flush()

                env = os.environ.copy()
                # Propagate remote settings to child processes so `train.py`/`eval.py` can fetch/sync checkpoints.
                if self.remote_enabled:
                    env["RCLONE_REMOTE"] = str(self.rclone_remote)
                    env["RCLONE_ROOT"] = str(self.rclone_root)
                    env["OWNER_ID"] = str(self.owner_id)

                # Handle "uv run python3" style commands: if python command contains spaces,
                # it's likely a compound command that needs shell execution.
                use_shell = " " in str(cmd[0]) if cmd else False
                if use_shell:
                    # Convert cmd list to shell string for compound commands like "uv run python3"
                    shell_cmd = " ".join(shlex.quote(str(arg)) for arg in cmd)
                    proc = self._run_with_tee_shell(cmd=shell_cmd, cwd=cwd, env=env, log_file=lf)
                else:
                    proc = self._run_with_tee(cmd=cmd, cwd=cwd, env=env, log_file=lf)
                exit_code = int(proc.returncode)
                ok = int(proc.returncode) == 0
                msg = "Success" if ok else f"Failed (exit {proc.returncode})"
                err_text = None if ok else f"exit_code={proc.returncode}"
        except KeyboardInterrupt:
            ok = False
            msg = "Interrupted"
            err_text = "KeyboardInterrupt"
        except Exception:
            ok = False
            msg = "Exception"
            err_text = traceback.format_exc()
        finally:
            # Best-effort: upload the per-node log to remote storage (e.g. Google Drive).
            # This makes failures debuggable even when workers are ephemeral/preempted.
            try:
                self._maybe_upload_node_log(node_id=node_id, log_path=log_path)
            except Exception:
                # Never fail the stage just because log upload failed.
                pass

        # Finalize status under global mutex, then release stage lock.
        try:
            with self.graph_mutex():
                self.remote_pull_graph()
                if self.graph_path.exists():
                    self._load_graph_file()
                if node_id in self.nodes:
                    n2 = self.nodes[node_id]
                    n2.last_exit_code = exit_code
                    n2.last_error = err_text
                    n2.updated_utc = _utc_ts()
                    if ok:
                        n2.status = "completed"
                    else:
                        n2.status = "interrupted" if msg == "Interrupted" else "failed"
                    self._persist()
        finally:
            self._release_remote_lock(lock_rel=stage_lock_rel)

        return ok, msg

    def _run_with_tee(
        self,
        *,
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        log_file,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a subprocess while:
        - streaming child's stdout -> parent stdout
        - streaming child's stderr -> parent stderr
        - appending both streams to `log_file` (stderr lines are prefixed for clarity)

        This is crucial for debugging failing work items: you see live output in the console,
        and you also have a durable per-node logfile.
        """
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        lock = threading.Lock()

        def pump(pipe, out_stream, *, is_stderr: bool) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    # Console
                    try:
                        out_stream.write(line)
                        out_stream.flush()
                    except Exception:
                        pass

                    # File (serialize writes across both threads)
                    try:
                        with lock:
                            if is_stderr:
                                log_file.write("[stderr] " + line)
                            else:
                                log_file.write(line)
                            log_file.flush()
                    except Exception:
                        pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=pump, args=(proc.stdout, sys.stdout), kwargs={"is_stderr": False}, daemon=True)
        t_err = threading.Thread(target=pump, args=(proc.stderr, sys.stderr), kwargs={"is_stderr": True}, daemon=True)
        t_out.start()
        t_err.start()

        rc = proc.wait()
        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="")

    def _run_with_tee_shell(
        self,
        *,
        cmd: str,
        cwd: str,
        env: dict[str, str],
        log_file,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a shell command (e.g., "uv run python3 script.py") while:
        - streaming child's stdout -> parent stdout
        - streaming child's stderr -> parent stderr
        - appending both streams to `log_file` (stderr lines are prefixed for clarity)
        """
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=True,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        lock = threading.Lock()

        def pump(pipe, out_stream, *, is_stderr: bool) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    # Console
                    try:
                        out_stream.write(line)
                        out_stream.flush()
                    except Exception:
                        pass

                    # File (serialize writes across both threads)
                    try:
                        with lock:
                            if is_stderr:
                                log_file.write("[stderr] " + line)
                            else:
                                log_file.write(line)
                            log_file.flush()
                    except Exception:
                        pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=pump, args=(proc.stdout, sys.stdout), kwargs={"is_stderr": False}, daemon=True)
        t_err = threading.Thread(target=pump, args=(proc.stderr, sys.stderr), kwargs={"is_stderr": True}, daemon=True)
        t_out.start()
        t_err.start()

        rc = proc.wait()
        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="")

    def _maybe_upload_node_log(self, *, node_id: str, log_path: Path) -> None:
        if not self.remote_enabled:
            return
        if not log_path.exists():
            return
        # Store under <remote_root>/logs/<safe_node_id>.log
        safe = self._safe_name(node_id)
        rel = f"logs/{safe}.log"
        # Ensure remote directory exists.
        self._rclone(["mkdir", self._remote_path("logs")], check=False)
        self._rclone(["copyto", str(log_path), self._remote_path(rel)], check=False)

