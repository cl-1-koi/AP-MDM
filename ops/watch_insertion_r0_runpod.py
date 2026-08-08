"""Sync, verify, and terminate one ordinary Sudoku insertion R0 pod."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import runpod, tomli

def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""): d.update(chunk)
    return d.hexdigest()

def now(): return datetime.now(timezone.utc).isoformat()

def atomic(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pod-id",required=True); p.add_argument("--endpoint",required=True)
    p.add_argument("--port",required=True); p.add_argument("--remote-dir",required=True)
    p.add_argument("--local-dir",required=True); p.add_argument("--source-commit",required=True)
    p.add_argument("--arm",required=True); p.add_argument("--max-seconds",type=int,default=1800)
    p.add_argument("--retain-rationale",default=None)
    a=p.parse_args(); local=Path(a.local_dir).resolve(); local.mkdir(parents=True,exist_ok=True)
    ssh_base=["ssh","-i","/home/ubuntu/.ssh/id_ed25519","-p",str(a.port),"-o","BatchMode=yes","-o","ConnectTimeout=15",a.endpoint]
    started=time.monotonic()
    state="unknown"
    while time.monotonic()-started <= a.max_seconds:
        command=f"if test -s {a.remote_dir}/completion.json; then echo complete; elif test -s {a.remote_dir}/failure.json; then echo failed; else echo running; fi; tail -n 1 {a.remote_dir}/train.log 2>/dev/null || true"
        result=subprocess.run(ssh_base+[command],text=True,capture_output=True)
        state=result.stdout.splitlines()[0] if result.returncode==0 and result.stdout else "unreachable"
        with (local/"watcher.jsonl").open("a") as h:
            h.write(json.dumps({"utc":now(),"state":state,"remote":result.stdout[-1000:]},sort_keys=True)+"\n")
        if state in {"complete","failed"}: break
        time.sleep(30)
    else: raise RuntimeError("watcher ceiling exceeded")
    remote_stderr = local / "sync_remote.stderr"
    with remote_stderr.open("wb") as stderr_handle:
        remote = subprocess.Popen(
            ssh_base + ["tar", "-C", a.remote_dir, "-cf", "-", "."],
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
        )
        assert remote.stdout is not None
        extracted = subprocess.run(
            ["tar", "-C", str(local), "-xf", "-"], stdin=remote.stdout
        )
        remote.stdout.close()
        remote_status = remote.wait()
    if extracted.returncode != 0 or remote_status != 0:
        raise RuntimeError(
            f"artifact sync failed: remote={remote_status}, local={extracted.returncode}"
        )
    closure={"pod_id":a.pod_id,"arm":a.arm,"source_commit":a.source_commit,"synced_utc":now(),"kind":state}
    if state=="complete":
        completion=json.loads((local/"completion.json").read_text())
        if completion["source_commit"]!=a.source_commit or completion["arm"]!=a.arm: raise RuntimeError("completion identity mismatch")
        checked=[]
        for entry in completion["files"]:
            path=local/entry["path"]
            if not path.is_file() or path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]: raise RuntimeError(f"artifact mismatch {entry['path']}")
            checked.append(entry["path"])
        closure.update({"completion_sha256":sha(local/"completion.json"),"verified_files":checked})
    else:
        failure=json.loads((local/"failure.json").read_text())
        if failure["source_commit"]!=a.source_commit or failure["arm"]!=a.arm: raise RuntimeError("failure identity mismatch")
        closure["failure"]=failure
        closure["verified_files"]=[{"path":str(x.relative_to(local)),"sha256":sha(x)} for x in local.rglob("*") if x.is_file()]
    atomic(local/"artifact_closure.json",closure)
    if a.retain_rationale:
        closure.update({"retained_utc":now(),"retain_rationale":a.retain_rationale})
        atomic(local/"artifact_closure.json",closure)
        return
    cfg=tomli.load(open("/home/ubuntu/.runpod/config.toml","rb")); runpod.api_key=cfg["default"]["api_key"]
    runpod.terminate_pod(a.pod_id)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        if not any(x.get("id")==a.pod_id for x in runpod.get_pods()):
            closure.update({"terminated_utc":now(),"inventory_absent_verified":True}); atomic(local/"artifact_closure.json",closure); return
        time.sleep(10)
    raise RuntimeError("pod remained in inventory")

if __name__=="__main__": main()
