"""Required online W&B logging for development experiments."""
from __future__ import annotations

import json
from pathlib import Path


def start_tracking(output: str | Path, *, project: str, name: str, config: dict):
    import wandb
    if not wandb.login():
        raise RuntimeError("W&B online login is required; refusing untracked experiment")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "wandb_run_id.txt"
    run_id = path.read_text().strip() if path.is_file() else wandb.util.generate_id()
    run = wandb.init(project=project, name=name, id=run_id, resume="allow", mode="online", dir=str(directory), config=config)
    if run is None or run.settings.mode != "online":
        raise RuntimeError("W&B did not start an online run")
    if not path.exists():
        path.write_text(run.id + "\n")
    (directory / "wandb_run.json").write_text(json.dumps({"id": run.id, "url": run.url, "project": project}, indent=2) + "\n")
    print("W&B:", run.url, flush=True)
    return run
