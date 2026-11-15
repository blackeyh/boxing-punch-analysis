"""Training progress reporting for the Claude Code status-line gauge.

Writes ~/.claude/train-progress.json, which ~/.claude/statusline-train.sh reads
every 2 s to draw a bar at the bottom of Claude Code:

    🥊 deep.py cnn seed 1  ███████░░░░░░░░  47%  12/26 min  ETA 14m

Usage:
    from progress import progress
    progress.start("deep.py", total=660)   # total work units (fits, epochs, ...)
    progress.tick(1, note="cnn seed 0")     # call as work completes
    progress.finish()
"""
import json
import os
import time
from pathlib import Path

# On the HPC, hpc.py sets PROGRESS_PATH to the run folder; that file is synced back
# here and copied into ~/.claude/train-progress.json for the local status line.
PATH = Path(os.environ.get("PROGRESS_PATH", Path.home() / ".claude" / "train-progress.json"))


class Progress:
    def __init__(self):
        self.active = False

    def start(self, label, total):
        self.active, self.label, self.total = True, label, float(total)
        self.done, self.note, self.t0 = 0.0, "", time.time()
        self._write("running")

    def tick(self, amount=1.0, note=None):
        if not self.active:
            return
        self.done = min(self.done + amount, self.total)
        if note is not None:
            self.note = note
        self._write("running")

    def finish(self, status="done"):
        if self.active:
            self.done = self.total if status == "done" else self.done
            self._write(status)
            self.active = False

    def _write(self, status):
        data = {"label": self.label, "note": self.note, "done": self.done, "total": self.total,
                "started": self.t0, "updated": time.time(), "pid": os.getpid(), "status": status}
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(PATH)  # atomic: the status line never reads a half-written file


progress = Progress()
