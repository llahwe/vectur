from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from experiment_manager import ExperimentManager


def _repo_code_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_python() -> str:
    """
    Smart default: use 'uv run python3' if uv is available and pyproject.toml exists.
    Otherwise fall back to 'python3'.
    """
    repo_root = _repo_code_dir().parent
    if shutil.which("uv") and (repo_root / "pyproject.toml").exists():
        return "uv run python3"
    return "python3"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiments via a persisted DAG work graph.")
    p.add_argument("--graph-path", type=str, default=str(_repo_code_dir() / "experiments" / "work_graph.json"))
    p.add_argument("--state-dir", type=str, default=str(_repo_code_dir() / ".experiment_manager"))
    p.add_argument("--python", type=str, default=_default_python())
    p.add_argument("--refresh", action="store_true", help="Pull latest graph from remote and re-load it.")
    p.add_argument("--init", action="store_true", help="Generate a fresh initial work graph (overwrites).")
    p.add_argument("--rclone-remote", type=str, default="gdrive:", help="rclone remote prefix, e.g. 'gdrive:'.")
    p.add_argument("--rclone-root", type=str, default="research/papers/vectur", help="Remote folder root, e.g. 'research/papers/vectur'.")
    p.add_argument("--owner-id", type=str, default=None, help="Override worker id used in remote lockfiles.")
    # IMPORTANT: stage locks are not refreshed while a stage is running, so TTL must be
    # longer than your longest expected stage runtime (e.g. multi-day training).
    p.add_argument("--lock-ttl-seconds", type=int, default=4 * 24 * 3600, help="Lock expiry time to avoid deadlocks.")
    p.add_argument("--no-remote", action="store_true", help="Disable remote state/locks even if env vars set.")

    p.add_argument("--max-stages", type=int, default=1, help="How many runnable stages to execute this run.")
    p.add_argument("--run-one", type=str, default=None, help="Run a specific node id regardless of scheduler choice.")

    p.add_argument("--manual-complete", type=str, default=None, help="Manually mark node completed.")
    p.add_argument("--manual-fail", type=str, default=None, help="Manually mark node failed.")
    p.add_argument("--manual-note", type=str, default=None, help="Note to attach to manual override.")
    p.add_argument("--list", action="store_true", help="List runnable stages and exit.")
    return p.parse_args()


def _banner(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def main() -> None:
    args = _parse_args()
    rclone_remote = None if bool(args.no_remote) else (str(args.rclone_remote) if args.rclone_remote else None)
    rclone_root = None if bool(args.no_remote) else (str(args.rclone_root) if args.rclone_root else None)
    mgr = ExperimentManager(
        graph_path=Path(args.graph_path),
        state_dir=Path(args.state_dir),
        python=str(args.python),
        rclone_remote=rclone_remote,
        rclone_root=rclone_root,
        owner_id=(str(args.owner_id) if args.owner_id else None),
        lock_ttl_seconds=int(args.lock_ttl_seconds),
    )
    
    _banner(f"Remote sync enabled: {mgr.remote_enabled}")
    if mgr.remote_enabled:
        _banner(f"  RCLONE_REMOTE: {mgr.rclone_remote}")
        _banner(f"  RCLONE_ROOT:   {mgr.rclone_root}")
        _banner(f"  Local State:   {mgr.state_dir}")

    if args.init:
        with mgr.graph_mutex():
            mgr.generate_initial_work_items(overwrite=True)
            mgr.load_graph()
        _banner(f"Wrote initial graph: {args.graph_path}")
        return

    # Load remote/local state first.
    mgr.load_graph()
    if args.refresh:
        mgr.load_graph()

    if args.manual_complete:
        mgr.set_status(str(args.manual_complete), "completed", note=args.manual_note)
        _banner(f"Manual override: {args.manual_complete} -> completed")
        return
    if args.manual_fail:
        mgr.set_status(str(args.manual_fail), "failed", note=args.manual_note)
        _banner(f"Manual override: {args.manual_fail} -> failed")
        return

    if args.list:
        mgr.load_graph()
        runnables = mgr.runnable_nodes()
        if not runnables:
            _banner("No runnable stages.")
            return
        _banner("Runnable stages:")
        for n in runnables:
            _banner(f"- {n.id}")
        return

    if args.run_one:
        node_id = str(args.run_one)
        _banner(f"Running Stage {node_id} ...")
        ok, msg = mgr.run_node(node_id)
        _banner("Success!" if ok else f"Failed: {msg}")
        return

    ran = 0
    while ran < int(args.max_stages):
        mgr.load_graph()
        runnables = mgr.runnable_nodes()
        if not runnables:
            if ran == 0:
                _banner("No runnable stages (either everything is done, or blocked by failures/manual nodes).")
            return

        ran_one = False
        attempted_unlocked = False
        for n in runnables:
            _banner(f"Running Stage {n.id} ...")
            ok, msg = mgr.run_node(n.id)
            if ok:
                _banner("Success!")
                ran_one = True
                break
            # If another worker holds the mutex, just try the next runnable node.
            if "Locked" in msg:
                _banner("Skipped (locked by another worker).")
                continue
            attempted_unlocked = True
            _banner(f"Failed: {msg}")
            # Continue: attempt other runnable stages in this invocation.
            continue
        if not ran_one:
            if attempted_unlocked:
                # We tried at least one stage and it failed; reload and see if any other
                # branches remain runnable now that the graph has been updated.
                continue
            _banner("No runnable stages (all currently locked by other workers).")
            return
        ran += 1


if __name__ == "__main__":
    main()

