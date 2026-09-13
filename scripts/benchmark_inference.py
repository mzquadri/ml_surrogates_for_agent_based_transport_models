"""Measure what one scenario costs the surrogate at inference.

The README opens on the claim that makes the whole approach worthwhile: MATSim
needs hours per scenario, and the surrogate "answers the same question in
seconds". That is the argument for building a surrogate at all, and nothing in
the repository measured it. This does.

It is a runtime measurement, not a result. Latency is a property of the machine
it ran on, so it is reported with the hardware named and is deliberately kept
out of scripts/verify_headline_results.py, which pins numbers that must
reproduce anywhere. Accuracy is not touched here: the thesis numbers come from
the recorded runs and this script does not recompute them.

    python scripts/benchmark_inference.py --dataloader <test_dl.pt>

The dataloader is a release asset, so this does not run in CI. Any scenario
works: the topology and five of the six feature columns are byte-identical
across scenarios, so cost does not vary between them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gnn.models.point_net_transf_gat import PointNetTransfGAT  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPO / "models" / "point_net_transf_gat_8th_trial_lower_dropout"
    / "trained_model" / "model.pth"
)


def load_model(checkpoint: Path) -> PointNetTransfGAT:
    """T8, the trial the thesis reports, loaded strictly.

    strict=True on purpose: a silently skipped tensor would leave part of the
    network at its initialisation and time a model nobody trained.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = PointNetTransfGAT(in_channels=5, out_channels=1, dropout=0.1, use_dropout=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def devices() -> list[str]:
    found = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        found.append("xpu")
    if torch.cuda.is_available():
        found.append("cuda")
    return found


def describe(device: str) -> str:
    if device == "xpu":
        return torch.xpu.get_device_properties(0).name
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def measure(model, graph, device: str, repeats: int, warmup: int) -> tuple[list[float], torch.Tensor]:
    target = torch.device(device)
    placed = model.to(target).eval()
    data = graph.clone().to(target)
    data.x = data.x.float()

    def sync() -> None:
        if device == "xpu":
            torch.xpu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):          # allocator warm, kernels compiled
            out = placed(data)
        sync()
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            out = placed(data)
            sync()
            timings.append((time.perf_counter() - started) * 1000)
    return timings, out.detach().float().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataloader", type=Path, required=True,
                        help="a saved list of Data objects, from the thesis-data-v1 release")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", type=Path, help="also write the measurement here")
    args = parser.parse_args()

    if not args.dataloader.is_file():
        print(f"No dataloader at {args.dataloader}.\n\n"
              "Fetch one with:\n"
              "  gh release download thesis-data-v1 "
              "--repo mzquadri/ml_surrogates_for_agent_based_transport_models \\n"
              "    --pattern '*5feat_seed42__dataloaders__test_dl.pt' --dir /tmp/large",
              file=sys.stderr)
        return 1

    graphs = torch.load(args.dataloader, map_location="cpu", weights_only=False)
    graphs = getattr(graphs, "dataset", graphs)
    graph = graphs[0]
    model = load_model(args.checkpoint)

    print(f"graph      {graph.num_nodes:,} nodes, {graph.edge_index.shape[1]:,} edges")
    print(f"model      {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"repeats    {args.repeats} timed, {args.warmup} warm-up\n")
    print(f"  {'device':6s} {'hardware':34s} {'median':>10s} {'min':>9s} {'max':>9s}")

    record: dict[str, dict] = {}
    outputs: dict[str, torch.Tensor] = {}
    for device in devices():
        timings, out = measure(model, graph, device, args.repeats, args.warmup)
        outputs[device] = out
        record[device] = {
            "hardware": describe(device),
            "median_ms": round(statistics.median(timings), 2),
            "min_ms": round(min(timings), 2),
            "max_ms": round(max(timings), 2),
        }
        print(f"  {device:6s} {describe(device)[:34]:34s} "
              f"{statistics.median(timings):9.1f}ms {min(timings):8.1f}ms {max(timings):8.1f}ms")

    # An accelerator that returns different numbers has not accelerated anything.
    baseline = outputs["cpu"]
    for device, out in outputs.items():
        if device == "cpu":
            continue
        gap = (baseline - out).abs().max().item()
        span = (baseline.max() - baseline.min()).item()
        record[device]["max_abs_diff_vs_cpu"] = float(f"{gap:.3e}")
        record[device]["relative_to_output_range"] = float(f"{gap / span:.3e}")
        print(f"\n  {device} agrees with cpu to {gap:.2e} "
              f"({gap / span:.1e} of the output range)")
        print(f"  speed-up over cpu: "
              f"{record['cpu']['median_ms'] / record[device]['median_ms']:.2f}x")

    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
