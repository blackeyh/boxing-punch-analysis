"""Run a project script on the MBZUAI HPC GPU, while THIS folder stays the source of truth.

    python hpc.py deep.py cnn --seeds 0 1 2            # auto-named run
    python hpc.py --name cnn_dur deep.py cnn --dur    # named run

What it does:
  1. PUSH   code + data from here -> HPC work dir (one-way; nothing is edited on the HPC)
  2. SUBMIT a Slurm job: 1 GPU on ws-ia, the pinned `punch` conda env (same versions as the Mac)
  3. SYNC   every few seconds, pull runs/<run>/ (log, progress, outputs) back here and
            feed the progress into the local status-line gauge
  4. FINISH final pull + job state, so runs/<run>/ here is a complete copy

Everything a run produces lands in ./runs/<run>/ on this Mac:
  output.log   stdout/stderr of the script      slurm.log   Slurm's own log
  job.sh       the exact job script submitted   meta.json   job id, node, state, timings
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
# Each workstream (main thread, background agents) MUST use its own HPC_REMOTE dir:
# the code push mirrors with --delete, so two workstreams sharing one dir would delete
# each other's files. GAUGE_PATH likewise gives each workstream its own status-line bar.
REMOTE = os.environ.get("HPC_REMOTE", "/l/users/<user>/punch2")   # punch/ has a stuck Lustre lock (incident 2)
HOST = "hpc"
# Keep-alives make a dead connection fail in ~45 s instead of hanging forever (seen after a
# laptop sleep: the sync loop froze inside ssh while the job had long finished).
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
       "-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home()}/.ssh/cm-%r@%h:%p", "-o", "ControlPersist=10m"]
CALL_TIMEOUT = 600   # hard limit (s) on any single ssh/rsync call
GAUGE = Path(os.environ.get("GAUGE_PATH", Path.home() / ".claude" / "train-progress.json"))
# video3d/raw/ (videos, per-frame pose dumps) lives on the HPC and is pulled back separately;
# excluding it also protects it from the --delete mirror.
PUSH_EXCLUDE = ["runs/", "__pycache__/", "*.log", ".claude/", "*.tmp", ".git/", ".DS_Store", "video3d/raw/", "videodet/raw/",
                "videodet/cache/"]


def sh(cmd, check=True, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=CALL_TIMEOUT)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip() if capture else ""


def remote(cmd, check=True):
    return sh(SSH + [HOST, cmd], check=check)


def rsync(src, dst, *extra):
    sh(["rsync", "-az", "-e", " ".join(SSH), *extra, src, dst])


def pull_safe(src, dst):
    """Sync a LIVE run folder back. Files can change mid-copy (progress.json is replaced
    every epoch), so skip temp files and treat rsync's 'vanished files' (24) and
    'partial transfer' (23) codes as fine: the next sync picks the files up. Network
    errors are reported but never kill the sync loop."""
    try:
        r = subprocess.run(["rsync", "-az", "-e", " ".join(SSH), "--exclude", "job.sh", "--exclude", "*.tmp", "--exclude", ".*.part",
                            src, dst], capture_output=True, text=True, timeout=CALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[hpc] sync timed out after {CALL_TIMEOUT}s, retrying next round", flush=True)
        return
    if r.returncode not in (0, 23, 24):
        print(f"[hpc] sync warning (rsync {r.returncode}), retrying next round: {r.stderr.strip()[:200]}", flush=True)


def job_script(run, cmd, args):
    """The job WRITES ONLY TO NODE-LOCAL DISK while it runs ($LOCAL_RUN), so training can never
    block on the shared Lustre storage (/l). A background loop publishes small snapshots of the
    log and progress to /l every 30 s (copy to a temp name, then rename), and everything is
    copied to /l once at the end. If /l stalls, only that copier waits; training does not.
    (Incident 2, 2026-11-12: a dead rsync left an unkillable Lustre lock that froze every job
    writing into the same folder.)"""
    shared = f"{REMOTE}/runs/{run}"
    return f"""#!/bin/bash
#SBATCH --job-name=punch-{run}
#SBATCH --partition={args.partition}
{"#SBATCH --gres=gpu:1" if not args.no_gpu else "# (CPU-only job: no GPU requested)"}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={shared}/slurm.log
source ~/apps/miniforge3/etc/profile.d/conda.sh
conda activate {args.env}
cd {REMOTE}
LOCAL_RUN=${{TMPDIR:-/tmp}}/punch_$SLURM_JOB_ID
mkdir -p $LOCAL_RUN
export PROGRESS_PATH=$LOCAL_RUN/progress.json
export RUN_DIR=$LOCAL_RUN
export PYTHONUNBUFFERED=1
publish() {{   # small atomic snapshots to shared storage; never blocks training
  for f in progress.json output.log; do
    [ -f $LOCAL_RUN/$f ] && timeout 60 cp $LOCAL_RUN/$f {shared}/.$f.part && timeout 60 mv {shared}/.$f.part {shared}/$f
  done
}}
( while true; do sleep 30; publish; done ) &
PUBLISHER=$!
echo "node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)  cpus: $SLURM_CPUS_PER_TASK  local: $LOCAL_RUN  start: $(date)"
python -u {cmd} > $LOCAL_RUN/output.log 2>&1
RC=$?
kill $PUBLISHER 2>/dev/null
timeout 600 cp -r $LOCAL_RUN/. {shared}/ && rm -rf $LOCAL_RUN
echo "exit code: $RC  end: $(date)"
"""


def final_state(j):
    """[state, exit_code, elapsed, node]. sacct's database is down on this cluster, so use
    scontrol (kept ~minutes after the job ends), falling back to the job's own slurm.log."""
    info = remote(f"scontrol show job {j['id']} 2>/dev/null", check=False)
    get = lambda k: next((w.split("=", 1)[1] for w in info.split() if w.startswith(k + "=")
                          and w.split("=", 1)[1] not in ("", "(null)")), "")
    state, code, elapsed = get("JobState"), get("ExitCode"), get("RunTime")
    node = next((w.split("=", 1)[1] for w in reversed(info.split())
                 if w.startswith("NodeList=") and w != "NodeList="), "")
    if not state:                                  # scontrol already forgot the job
        log = (j["ldir"] / "slurm.log")
        line = next((l for l in log.read_text().splitlines() if l.startswith("exit code:")), "") if log.exists() else ""
        rc = line.split()[2] if line else ""
        state = "COMPLETED" if rc == "0" else ("FAILED" if rc else "UNKNOWN")
        code = rc
    return [state, code, elapsed, node]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="run name (default: script + timestamp)")
    ap.add_argument("--sweep", help="file with one run per line: `name | script args...` (parallel jobs)")
    ap.add_argument("--partition", default="ws-ia")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--no-gpu", action="store_true", help="CPU-only job (no --gres=gpu:1)")
    ap.add_argument("--env", default="punch", help="conda env on the HPC (default: punch)")
    ap.add_argument("--interval", type=float, default=5, help="seconds between syncs")
    ap.add_argument("--attach", nargs="+", metavar="RUN",
                    help="re-attach the sync loop to already-submitted runs (names under runs/)")
    ap.add_argument("script", nargs=argparse.REMAINDER, help="script and its arguments")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.attach:
        jobs = []
        for run in args.attach:
            ldir = LOCAL / "runs" / run
            meta = json.loads((ldir / "meta.json").read_text())
            jobs.append({"run": run, "rdir": meta["remote_dir"], "ldir": ldir, "id": meta["job_id"],
                         "meta": meta, "state": None, "done": False})
            print(f"[hpc] attached to job {meta['job_id']}: {run}", flush=True)
        return sync_loop(jobs, args)
    if args.sweep:
        specs = []
        for line in Path(args.sweep).read_text().splitlines():
            if line.strip() and not line.strip().startswith("#"):
                name, cmd = (x.strip() for x in line.split("|", 1))
                specs.append((f"{name}-{stamp}", cmd))
    elif args.script:
        cmd = " ".join(shlex.quote(a) for a in args.script)
        specs = [(args.name or f"{Path(args.script[0]).stem}-{stamp}", cmd)]
    else:
        ap.error("give a script to run (e.g. deep.py cnn --seeds 0) or --sweep FILE")

    # 1. push code + data (one-way mirror)
    print(f"[hpc] pushing code + data -> {HOST}:{REMOTE}", flush=True)
    excl = sum((["--exclude", e] for e in PUSH_EXCLUDE), [])
    rsync(f"{LOCAL}/", f"{HOST}:{REMOTE}/", "--delete", *excl)

    # 2. submit one GPU job per run
    jobs = []
    for run, cmd in specs:
        rdir, ldir = f"{REMOTE}/runs/{run}", LOCAL / "runs" / run
        ldir.mkdir(parents=True, exist_ok=True)
        (ldir / "job.sh").write_text(job_script(run, cmd, args))
        remote(f"mkdir -p {rdir}")
        rsync(str(ldir / "job.sh"), f"{HOST}:{rdir}/job.sh")
        job_id = remote(f"sbatch --parsable {rdir}/job.sh").split(";")[0]
        meta = {"run": run, "command": cmd, "job_id": job_id, "submitted": stamp, "remote_dir": rdir}
        (ldir / "meta.json").write_text(json.dumps(meta, indent=2))
        jobs.append({"run": run, "rdir": rdir, "ldir": ldir, "id": job_id, "meta": meta, "state": None, "done": False})
        print(f"[hpc] submitted job {job_id}: {run}  ->  {cmd}", flush=True)
    sync_loop(jobs, args)


def sync_loop(jobs, args):
    # 3. sync loop: pull every run folder, combine progress into ONE gauge
    t0 = time.time()

    def pull(j):
        pull_safe(f"{HOST}:{j['rdir']}/", f"{j['ldir']}/")

    STALL_MIN = 10   # a RUNNING job whose progress hasn't moved for this long is flagged as stalled

    def gauge(status="running"):
        done = total = 0.0
        stalled = []
        for j in jobs:
            p = j["ldir"] / "progress.json"
            try:
                d = json.loads(p.read_text()); done += d["done"]; total += d["total"]
                age = (time.time() - d["updated"]) / 60
                if j["state"] == "RUNNING" and not j["done"] and age > STALL_MIN:
                    stalled.append(j["run"])
                    if not j.get("warned"):
                        print(f"[hpc] {datetime.now():%H:%M:%S} WARNING {j['run']}: no progress for {age:.0f} min "
                              f"(job {j['id']} may be hung; check it or cancel with: ssh hpc scancel {j['id']})", flush=True)
                        j["warned"] = True
            except (OSError, ValueError, KeyError):
                total += 1                        # not started yet: count as 1 unit of unknown work
        n_run = sum(j["state"] == "RUNNING" for j in jobs); n_done = sum(j["done"] for j in jobs)
        label = f"HPC {len(jobs)} job{'s' * (len(jobs) > 1)}" if len(jobs) > 1 else f"HPC {jobs[0]['run']}"
        note = f"{n_done} done, {n_run} running" if len(jobs) > 1 else ""
        if stalled:
            note = f"STALLED: {', '.join(stalled)} | " + note
        d = {"label": label, "note": note, "done": done, "total": max(total, 1e-9), "started": t0,
             "updated": time.time(), "pid": os.getpid(), "status": status}
        tmp = GAUGE.with_suffix(".tmp"); tmp.write_text(json.dumps(d)); tmp.replace(GAUGE)

    while not all(j["done"] for j in jobs):
        ids = ",".join(j["id"] for j in jobs if not j["done"])
        try:
            r = subprocess.run(SSH + [HOST, f"squeue -h -j {ids} -o '%i %T' 2>/dev/null; echo __OK__"],
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            r = subprocess.CompletedProcess([], 1, "", "timeout")
        if r.returncode != 0 or "__OK__" not in r.stdout:   # ssh hiccup: don't mistake it for "finished"
            print(f"[hpc] {datetime.now():%H:%M:%S} couldn't reach the HPC, retrying", flush=True)
            time.sleep(args.interval); continue
        q = dict(line.split(maxsplit=1) for line in r.stdout.splitlines()
                 if line.strip() and line.strip() != "__OK__")
        for j in jobs:
            if j["done"]:
                continue
            st = q.get(j["id"])
            if st is None:                        # left the queue
                time.sleep(1); pull(j)
                parts = final_state(j)
                j["meta"].update(state=parts[0], exit_code=parts[1], elapsed=parts[2], node=parts[3],
                                 finished=f"{datetime.now():%Y%m%d-%H%M%S}")
                j["done"], j["state"] = True, parts[0]
                print(f"[hpc] {datetime.now():%H:%M:%S} {j['run']}: {parts[0]} (exit {parts[1]}, {parts[2]} on {parts[3]})", flush=True)
            else:
                if st != j["state"]:
                    print(f"[hpc] {datetime.now():%H:%M:%S} {j['run']}: {st}", flush=True)
                    j["state"] = st; j["meta"]["state"] = st
                if st == "RUNNING":
                    pull(j)
            (j["ldir"] / "meta.json").write_text(json.dumps(j["meta"], indent=2))
        gauge()
        if not all(j["done"] for j in jobs):
            time.sleep(args.interval)

    # 4. finish
    ok = all(j["state"] == "COMPLETED" for j in jobs)
    gauge("done" if ok else "stopped")
    for j in jobs:
        print(f"[hpc] {j['run']}: {j['state']}  ->  {j['ldir']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
