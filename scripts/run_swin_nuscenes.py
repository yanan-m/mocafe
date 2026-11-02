"""Swin-T MoCaFe / p-MoCaFe on nuScenes: One North (source) -> Boston Seaport (target).

Stages, each resumable (finished work is skipped):

    run_swin_nuscenes.py --data DATA --out OUT pretrain --gpu 0
    run_swin_nuscenes.py --data DATA --out OUT select --gpus 0,1,2,3
    run_swin_nuscenes.py --data DATA --out OUT adapt --gpus 0,1,2,3
    run_swin_nuscenes.py --data DATA --out OUT summarize

``DATA`` holds the crops and the manifests of the release README plus ``source_train.csv``
(``image``, ``class_id``: labelled One North frames, disjoint from the public set).

1. ``pretrain`` fine-tunes an ImageNet torchvision Swin-T on ``source_train.csv``. The paper does
   not give the Swin-T source recipe; we use AdamW (lr 5e-4, weight decay 0.05), a 5-epoch
   warmup and cosine decay. Swin-T passes 65-75 % Boston accuracy within its first ~2,500
   steps, so weights are saved densely early on (every 20 steps up to 600, every 50 up to 1,500,
   every 250 up to 5,000) and after every epoch.
2. ``select`` evaluates every saved checkpoint on ``target_test.csv`` and picks the ones closest
   to the paper's pre-trained accuracies (65 %, 70 %, 75 %).
3. ``adapt`` runs MoCaFe and p-MoCaFe from each start with seeds 0, 1 and 2 through the release
   API, one job per GPU, and stores each run's frame selection.
4. ``summarize`` checks that each MoCaFe / p-MoCaFe pair trained on the same frames and
   pseudo-labels and writes ``results.json`` and ``results.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import mocafe
from mocafe import _data, _models, _selection, _training

SOURCE_TRAIN = "source_train.csv"
TARGETS = (0.65, 0.70, 0.75)
SEEDS = (0, 1, 2)
METHODS = ("mocafe", "p-mocafe")
NUM_CLASSES = 10
SNAPSHOT_STEPS = (*range(20, 600, 20), *range(600, 1500, 50), *range(1500, 5001, 250))
RETRIES = 2
ADAPT = {"num_samples": 32000, "epochs": 50, "batch_size": 32, "num_workers": 8}
PAPER = {  # Table 1, Swin-T, Boston Seaport
    0.75: {"mocafe": "89.60 ± 0.11", "p-mocafe": "90.19 ± 0.17"},
    0.70: {"mocafe": "89.34 ± 0.15", "p-mocafe": "90.07 ± 0.18"},
    0.65: {"mocafe": "89.27 ± 0.16", "p-mocafe": "90.02 ± 0.12"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_atomic(model: nn.Module, path: Path, **extra) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _models.save_model(model, tmp, **extra)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def start_name(target: float) -> str:
    return f"start_{round(target * 100):03d}"


def run_dir(out: Path, target: float, seed: int, method: str) -> Path:
    return out / "adapt" / start_name(target) / f"seed{seed}" / method


# ----------------------------------------------------------------------------- pretraining


def pretrain(args) -> None:
    src = args.out / "source"
    src.mkdir(parents=True, exist_ok=True)
    rows = _data.read_manifest(args.data / SOURCE_TRAIN, ("image", "class_id"))
    labels = _data.class_ids(rows, NUM_CLASSES, args.data / SOURCE_TRAIN)
    settings = {
        "backbone": _models.SWIN_T,
        "init": "torchvision Swin_T_Weights.IMAGENET1K_V1",
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": _training.SWIN_WEIGHT_DECAY,
        "warmup_epochs": _training.SWIN_WARMUP_EPOCHS,
        "schedule": "per-step linear warmup from lr*1e-3, cosine decay to lr*1e-2",
        "clip_grad_norm": _training.SWIN_CLIP_GRAD,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "augmentation": "RandomResizedCrop(224), RandomHorizontalFlip",
        "amp": "bfloat16 autocast",
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "source_train_sha256": sha256(args.data / SOURCE_TRAIN),
        "frames": len(rows),
    }
    meta = src / "pretrain.json"
    if meta.exists() and json.loads(meta.read_text())["settings"] != settings:
        sys.exit(f"{meta} was written with different settings; use a new --out")
    write_json(meta, {"settings": settings, "complete": False})

    device = torch.device("cuda")
    _training.seed_everything(args.seed)
    model = _models.build_model(_models.SWIN_T, NUM_CLASSES, imagenet_init=True).to(device)
    dataset = _training.CropDataset(
        args.data, [r["image"] for r in rows], labels, transform=_training.TRAIN_TRANSFORM
    )
    loader = _training.make_loader(
        dataset, args.batch_size, args.num_workers, shuffle=True, seed=args.seed
    )
    optimizer, scheduler, clip = _training.make_optimizer(model, args.lr, args.epochs, len(loader))
    start, history = 0, []
    last = src / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = state["epoch"], state["history"]
        print(f"resuming after epoch {start}", flush=True)

    step = start * len(loader)
    base_seed = loader.generator.initial_seed()
    for epoch in range(start + 1, args.epochs + 1):
        torch.manual_seed(base_seed + epoch)
        loader.generator.manual_seed(base_seed + epoch)
        model.train()
        total, correct, seen, tic = 0.0, 0, 0, time.time()
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with _training._autocast(device):
                logits = model(inputs)
            loss = F.cross_entropy(logits.float(), targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            step += 1
            total += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.size(0)
            if step in SNAPSHOT_STEPS:
                save_atomic(model, src / f"step_{step:06d}.pt", step=step)
            if step % 500 == 0:
                print(
                    f"epoch {epoch} step {step}: loss {total / seen:.4f} "
                    f"acc {correct / seen:.4f} lr {optimizer.param_groups[0]['lr']:.3g} "
                    f"{seen / (time.time() - tic):.0f} img/s",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "train_loss": total / seen,
                "train_acc": correct / seen,
                "minutes": (time.time() - tic) / 60,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        save_atomic(model, src / f"epoch_{epoch:03d}.pt", epoch=epoch)
        tmp = last.with_name("last.pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
            },
            tmp,
        )
        os.replace(tmp, last)
        with open(src / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)
    write_json(meta, {"settings": settings, "complete": True})


# ----------------------------------------------------------------------------- GPU queue


def run_queue(jobs: list[tuple[str, list[str]]], gpus: list[str], logs: Path, done) -> list:
    """Run ``[(name, argv)]`` one per GPU; retry failures; return the jobs that still failed."""
    logs.mkdir(parents=True, exist_ok=True)
    queue = [(name, argv, 0) for name, argv in jobs if not done(name)]
    running, failed = {}, []
    print(f"{len(queue)} job(s) on GPUs {','.join(gpus)}", flush=True)
    while queue or running:
        for gpu in [g for g in gpus if g not in running]:
            if not queue:
                break
            name, argv, attempt = queue.pop(0)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log = open(logs / f"{name}_try{attempt}_{stamp}.log", "w")  # noqa: SIM115
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (proc, log, name, argv, attempt)
            print(f"{stamp} start {name} (try {attempt}) on GPU {gpu}", flush=True)
        time.sleep(5)
        for gpu, (proc, log, name, argv, attempt) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[gpu]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"{stamp} end {name}: exit {code}", flush=True)
            if code == 0 and done(name):
                continue
            if attempt < RETRIES:
                queue.append((name, argv, attempt + 1))
            else:
                failed.append(name)
    return failed


def base_argv(args) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--data", str(args.data)]


# ----------------------------------------------------------------------------- selection


def snapshots(src: Path) -> list[Path]:
    return sorted(src.glob("step_*.pt")) + sorted(src.glob("epoch_*.pt"))


def eval_record(src: Path, ckpt: Path) -> Path:
    return src / "eval" / f"{ckpt.stem}.json"


def is_evaluated(src: Path, ckpt: Path) -> bool:
    record = eval_record(src, ckpt)
    if not record.exists():
        return False
    r, st = json.loads(record.read_text()), ckpt.stat()
    return r["size"] == st.st_size and r["mtime_ns"] == st.st_mtime_ns


def eval_job(args) -> None:
    ckpt = Path(args.checkpoint)
    st = ckpt.stat()
    test = _data.read_manifest(args.data / _data.TARGET_TEST, ("image", "class_id"))
    labels = np.array(_data.class_ids(test, NUM_CLASSES, args.data / _data.TARGET_TEST))
    model = _training.load_model(ckpt)
    dataset = _training.CropDataset(args.data, [r["image"] for r in test], labels)
    loader = _training.make_loader(dataset, 128, 8)
    pred = _training.predict(model, loader, torch.device("cuda")).argmax(1)
    record = eval_record(ckpt.parent, ckpt)
    record.parent.mkdir(exist_ok=True)
    write_json(
        record,
        {
            "checkpoint": ckpt.name,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "accuracy": float((pred == labels).mean()),
            "per_class": [float((pred[labels == c] == c).mean()) for c in range(NUM_CLASSES)],
        },
    )


def select(args) -> None:
    src = args.out / "source"
    complete = json.loads((src / "pretrain.json").read_text())["complete"]
    if not (complete or args.eval_only):
        sys.exit("pretraining is not complete; use --eval-only to evaluate saved checkpoints")
    ckpts = {c.stem: c for c in snapshots(src)}
    jobs = [(stem, [*base_argv(args), "eval-job", str(c)]) for stem, c in ckpts.items()]
    failed = run_queue(
        jobs, args.gpus, args.out / "logs" / "select", lambda s: is_evaluated(src, ckpts[s])
    )
    if failed:
        sys.exit(f"evaluation failed: {failed}")
    records = [json.loads(eval_record(src, c).read_text()) for c in ckpts.values()]
    with open(src / "checkpoint_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "accuracy"])
        writer.writerows([r["checkpoint"], f"{r['accuracy']:.6f}"] for r in records)
    for r in records:
        print(f"{r['checkpoint']:>18} {100 * r['accuracy']:.2f}")
    if args.eval_only:
        return
    selected = {"test_manifest_sha256": sha256(args.data / _data.TARGET_TEST), "starts": {}}
    for target in TARGETS:
        best = min(records, key=lambda r: abs(r["accuracy"] - target))
        path = src / best["checkpoint"]
        selected["starts"][start_name(target)] = {
            "target": target,
            "checkpoint": str(path),
            "accuracy": best["accuracy"],
            "sha256": sha256(path),
        }
    picks = [s["checkpoint"] for s in selected["starts"].values()]
    if len(set(picks)) != len(picks):
        sys.exit(f"two targets share a checkpoint: {picks}")
    write_json(src / "selected.json", selected)
    print(json.dumps(selected, indent=2))


# ----------------------------------------------------------------------------- adaptation


def adapt_done(out: Path, target: float, seed: int, method: str) -> bool:
    path = run_dir(out, target, seed, method) / "run.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "done"


def adapt_job(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    selected = json.loads((args.out / "source" / "selected.json").read_text())
    start = selected["starts"][start_name(args.target)]
    ckpt = Path(start["checkpoint"])
    if sha256(ckpt) != start["sha256"]:
        sys.exit(f"{ckpt} changed since selection")
    out = run_dir(args.out, args.target, args.seed, args.method)
    out.mkdir(parents=True, exist_ok=True)

    select_frames = _selection.select_frames

    def recording(*a, **k):
        s = select_frames(*a, **k)
        np.savez(out / "selection.npz", indices=s.indices, labels=s.labels, beta=s.beta)
        return s

    _selection.select_frames = recording
    fn = mocafe.mocafe if args.method == "mocafe" else mocafe.p_mocafe
    tic = time.time()
    metrics = fn(ckpt, args.data, out, seed=args.seed, **ADAPT)
    write_json(
        out / "run.json",
        {
            "status": "done",
            "target": args.target,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": start["sha256"],
            "model_sha256": sha256(out / "model.pt"),
            "elapsed_seconds": time.time() - tic,
            "gpu": torch.cuda.get_device_name(),
            "metrics": metrics,
        },
    )


def adapt(args) -> None:
    if not (args.out / "source" / "selected.json").exists():
        sys.exit("run `select` first")
    jobs, keys = [], {}
    for target in TARGETS:
        for seed in args.seeds:
            for method in METHODS:
                name = f"{start_name(target)}_seed{seed}_{method}"
                keys[name] = (target, seed, method)
                argv = [
                    *base_argv(args),
                    "--out",
                    str(args.out),
                    "adapt-job",
                    "--target",
                    str(target),
                    "--seed",
                    str(seed),
                    "--method",
                    method,
                ]
                jobs.append((name, argv))
    failed = run_queue(
        jobs, args.gpus, args.out / "logs" / "adapt", lambda n: adapt_done(args.out, *keys[n])
    )
    if failed:
        sys.exit(f"adaptation failed: {failed}")


# ----------------------------------------------------------------------------- summary


def mean_sd(values) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(args) -> None:
    selected = json.loads((args.out / "source" / "selected.json").read_text())
    pretrain_meta = json.loads((args.out / "source" / "pretrain.json").read_text())
    errors, cells = [], {}

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    for target in TARGETS:
        start = selected["starts"][start_name(target)]
        for seed in args.seeds:
            sel = {}
            for method in METHODS:
                d = run_dir(args.out, target, seed, method)
                name = f"{start_name(target)} seed{seed} {method}"
                if not adapt_done(args.out, target, seed, method):
                    errors.append(f"{name}: missing")
                    continue
                run = json.loads((d / "run.json").read_text())
                m = json.loads((d / "metrics.json").read_text())
                check(run["checkpoint_sha256"] == start["sha256"], f"{name}: checkpoint hash")
                check(run["model_sha256"] == sha256(d / "model.pt"), f"{name}: model hash")
                check(m["backbone"] == _models.SWIN_T, f"{name}: backbone")
                check(m["method"] == method and m["seed"] == seed, f"{name}: method/seed")
                check(m["lr"] == 1e-5, f"{name}: lr {m['lr']}")
                check(all(m[k] == v for k, v in ADAPT.items() if k in m), f"{name}: settings")
                # Selection and adaptation evaluate the same checkpoint on the same test set;
                # a few-frame difference is expected under bfloat16 GPU non-determinism.
                check(
                    abs(m["source_accuracy"] - start["accuracy"]) < 1e-4,
                    f"{name}: source accuracy {m['source_accuracy']} != {start['accuracy']}",
                )
                if method == "p-mocafe":
                    check(m.get("public_frames") == 8000 and m.get("k") == 45, f"{name}: v, k")
                sel[method] = np.load(d / "selection.npz")
                cells[target, seed, method] = {
                    "accuracy": 100 * m["accuracy"],
                    "source_accuracy": 100 * m["source_accuracy"],
                    "minutes": run["elapsed_seconds"] / 60,
                }
            if len(sel) == 2:
                a, b = sel["mocafe"], sel["p-mocafe"]
                check(
                    np.array_equal(a["indices"], b["indices"])
                    and np.array_equal(a["labels"], b["labels"])
                    and np.array_equal(a["beta"], b["beta"]),
                    f"{start_name(target)} seed{seed}: methods sampled different frames",
                )
    if errors:
        print("CHECK FAILURES:\n" + "\n".join(errors))
        sys.exit(1)

    test_sha = sha256(args.data / _data.TARGET_TEST)
    results = {
        "backbone": _models.SWIN_T,
        "seeds": list(args.seeds),
        "adapt": {**ADAPT, "lr": 1e-5, "eta": _selection.ETA},
        "pretrain": pretrain_meta["settings"],
        "provenance": {
            "test_manifest_sha256": test_sha,
            "adapt_manifest_sha256": sha256(args.data / _data.TARGET_ADAPT),
            "public_manifest_sha256": sha256(args.data / _data.PUBLIC),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "python": platform.python_version(),
        },
        "starts": {},
    }
    check(test_sha == selected["test_manifest_sha256"], "test manifest changed since selection")
    lines = [
        "| Source accuracy | MoCaFe | p-MoCaFe | p-MoCaFe − MoCaFe (paired) "
        "| Paper MoCaFe | Paper p-MoCaFe |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target in sorted(TARGETS):
        start = selected["starts"][start_name(target)]
        res = {
            "target": target,
            "checkpoint": Path(start["checkpoint"]).name,
            "source_accuracy": 100 * start["accuracy"],
        }
        for method in METHODS:
            v = [cells[target, s, method]["accuracy"] for s in args.seeds]
            mu, sd = mean_sd(v)
            res[method] = {"seeds": v, "mean": mu, "sd": sd}
            res[method]["minutes"] = float(
                np.mean([cells[target, s, method]["minutes"] for s in args.seeds])
            )
        diff = [p - m for p, m in zip(res["p-mocafe"]["seeds"], res["mocafe"]["seeds"])]
        mu, sd = mean_sd(diff)
        res["paired_difference"] = {"seeds": diff, "mean": mu, "sd": sd}
        results["starts"][start_name(target)] = res
        lines.append(
            f"| {res['source_accuracy']:.2f} | "
            f"{res['mocafe']['mean']:.2f} ± {res['mocafe']['sd']:.2f} | "
            f"{res['p-mocafe']['mean']:.2f} ± {res['p-mocafe']['sd']:.2f} | "
            f"{mu:+.2f} ± {sd:.2f} | {PAPER[target]['mocafe']} | {PAPER[target]['p-mocafe']} |"
        )
    lines += [
        "",
        "| Source accuracy | Checkpoint | Method | "
        + " | ".join(f"Seed {s}" for s in args.seeds)
        + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in args.seeds) + " |",
    ]
    for target in sorted(TARGETS):
        res = results["starts"][start_name(target)]
        for method in METHODS:
            lines.append(
                f"| {res['source_accuracy']:.2f} | {res['checkpoint']} | {method} | "
                + " | ".join(f"{x:.2f}" for x in res[method]["seeds"])
                + " |"
            )
    text = "\n".join(lines)
    print("checks: OK\n" + text)
    write_json(args.out / "results.json", results)
    (args.out / "results.md").write_text(text + "\n")


# ----------------------------------------------------------------------------- CLI


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="crops and CSV manifests")
    parser.add_argument("--out", type=Path, help="experiment directory")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("pretrain", help="Swin-T source training on source_train.csv")
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("select", help="evaluate checkpoints and choose the three starts")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--eval-only", action="store_true", help="evaluate without choosing")

    p = sub.add_parser("adapt", help="MoCaFe and p-MoCaFe from every start and seed")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seeds", default="0,1,2")

    p = sub.add_parser("summarize", help="verify the runs and write the result tables")
    p.add_argument("--seeds", default="0,1,2")

    p = sub.add_parser("eval-job")
    p.add_argument("checkpoint")

    p = sub.add_parser("adapt-job")
    p.add_argument("--target", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", choices=METHODS, required=True)

    args = parser.parse_args(argv)
    args.data = args.data.resolve()
    if args.stage != "eval-job":
        if args.out is None:
            parser.error("--out is required")
        args.out = args.out.resolve()
    if hasattr(args, "gpus"):
        args.gpus = args.gpus.split(",")
    if hasattr(args, "seeds"):
        args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.stage == "pretrain":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    stages = {
        "pretrain": pretrain,
        "select": select,
        "adapt": adapt,
        "summarize": summarize,
        "eval-job": eval_job,
        "adapt-job": adapt_job,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
