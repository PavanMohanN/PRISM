"""PRISM Phase 1 -- result provenance + multi-seed harness.

Every number in the paper must trace to ONE canonical record. This eliminates,
by construction, the mismatches the review flagged:
  * Table 8 / Figure 8 disagree on data source (item 28)
  * Figure 10 caption contradicts Table 9 (item 26)
  * Table 5 vs Table 7 cycle errors differ (item 27)
and enforces multi-seed reporting (item 16) and reproducibility metadata (item 22).

Usage
-----
    man = ResultManifest("results/manifest.json")
    man.run_seeds(experiment="phase4", task="slcp", method="PRISM",
                  fn=lambda seed: train_and_eval(seed), seeds=[0,1,2,3,4],
                  config=cfg)
    man.save()
    # later, tables/figures READ from the manifest, never from ad-hoc vars:
    val = man.agg("phase4", "slcp", "PRISM", "c2st")   # -> (mean, std, n)

Records are keyed by (experiment, task, method, seed, metric) with a config hash,
git hash, dtype, and timestamp attached, so a value can never appear twice with
two different numbers.
"""
from __future__ import annotations

import json
import time
import hashlib
import subprocess
import statistics
from dataclasses import dataclass, field, asdict
from typing import Callable


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _config_hash(config: dict | None) -> str:
    if not config:
        return "noconfig"
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


@dataclass
class Record:
    experiment: str
    task: str
    method: str
    seed: int
    metrics: dict
    config_hash: str = "noconfig"
    git_hash: str = field(default_factory=_git_hash)
    dtype: str = "float32"
    timestamp: float = field(default_factory=time.time)

    def key(self):
        return (self.experiment, self.task, self.method, self.seed)


class ResultManifest:
    def __init__(self, path: str):
        self.path = path
        self.records: dict = {}

    # ---- write ----
    def record(self, experiment, task, method, seed, metrics,
               config=None, dtype="float32"):
        rec = Record(experiment, task, method, int(seed), dict(metrics),
                     _config_hash(config), _git_hash(), dtype)
        self.records[rec.key()] = rec
        return rec

    def run_seeds(self, experiment, task, method, fn: Callable[[int], dict],
                  seeds, config=None, dtype="float32"):
        """Run fn(seed)->metrics for each seed, recording every result."""
        for s in seeds:
            metrics = fn(s)
            self.record(experiment, task, method, s, metrics, config, dtype)
        return self.agg_all(experiment, task, method)

    # ---- read (tables/figures use ONLY these) ----
    def _seeds_for(self, experiment, task, method):
        return [r for k, r in self.records.items()
                if k[:3] == (experiment, task, method)]

    def agg(self, experiment, task, method, metric):
        vals = [r.metrics[metric] for r in self._seeds_for(experiment, task, method)
                if metric in r.metrics]
        if not vals:
            raise KeyError(f"no records for {(experiment, task, method, metric)}")
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, std, len(vals)

    def agg_all(self, experiment, task, method):
        recs = self._seeds_for(experiment, task, method)
        metrics = sorted({m for r in recs for m in r.metrics})
        return {m: self.agg(experiment, task, method, m) for m in metrics}

    def fmt(self, experiment, task, method, metric, prec=3):
        """LaTeX-ready 'mean $\\pm$ std' -- the ONLY way tables render a value."""
        mean, std, n = self.agg(experiment, task, method, metric)
        if n > 1:
            return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"
        return f"{mean:.{prec}f}"

    # ---- persistence ----
    def save(self):
        import os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        blob = {"|".join(map(str, k)): asdict(v) for k, v in self.records.items()}
        with open(self.path, "w") as f:
            json.dump(blob, f, indent=2)
        return self.path

    def load(self):
        with open(self.path) as f:
            blob = json.load(f)
        self.records = {}
        for _, v in blob.items():
            rec = Record(**v)
            self.records[rec.key()] = rec
        return self


if __name__ == "__main__":
    import tempfile, os
    tmp = os.path.join(tempfile.mkdtemp(), "manifest.json")
    man = ResultManifest(tmp)

    # simulate a 5-seed experiment
    def fake_eval(seed):
        import random
        random.seed(seed)
        return {"c2st": 0.55 + 0.01 * random.random(),
                "ece": 0.02 + 0.005 * random.random()}

    agg = man.run_seeds("phase4", "slcp", "PRISM", fake_eval,
                        seeds=[0, 1, 2, 3, 4], config={"lr": 3e-4})
    man.save()
    reloaded = ResultManifest(tmp).load()
    mean, std, n = reloaded.agg("phase4", "slcp", "PRISM", "c2st")
    print(f"aggregated c2st over {n} seeds: {mean:.4f} +/- {std:.4f}")
    print("LaTeX cell:", reloaded.fmt("phase4", "slcp", "PRISM", "c2st"))
    print("round-trip save/load OK:", len(reloaded.records) == 5)
