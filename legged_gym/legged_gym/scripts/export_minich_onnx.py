"""Export a checkpoint-matched Mini Cheetah HIM JIT controller to ONNX on CPU.

Requires torch, onnx and onnxruntime. No Isaac Gym simulation is needed.
The paired model_<iteration>.jit must already exist in exported/policies.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from rsl_rl.modules import HIMActorCritic


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    checkpoints = list(run.glob("model_*.pt"))
    checkpoint = (max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
                  if args.checkpoint == -1 else run / f"model_{args.checkpoint}.pt")
    jit_path = run / "exported/policies" / (checkpoint.stem + ".jit")
    output = (args.output or run / "exported/policies/policy.onnx").resolve()

    torch.set_num_threads(1)
    saved = torch.load(str(checkpoint), map_location="cpu")
    state = saved["model_state_dict"]
    # This exporter is deliberately scoped to the current 6 x 45 Mini Cheetah HIM policy.
    reference = HIMActorCritic(270, 238, 45, 12, activation="elu").cpu().eval()
    reference.load_state_dict(state, strict=True)
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    for name, value in model.state_dict().items():
        source_name = (name.replace("estimator.", "estimator.encoder.", 1)
                       if name.startswith("estimator.") else name)
        if not torch.equal(value, state[source_name]):
            raise ValueError(f"JIT does not match checkpoint: {name}")
    expected_keys = {k for k in state if k.startswith(("actor.", "estimator.encoder."))}
    actual_keys = {k.replace("estimator.", "estimator.encoder.", 1)
                   if k.startswith("estimator.") else k for k in model.state_dict()}
    if actual_keys != expected_keys:
        raise ValueError("JIT must contain exactly the actor and estimator encoder weights")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 270), str(output), export_params=True,
        opset_version=11, do_constant_folding=True,
        input_names=["obs"], output_names=["actions"],
    )
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(graph, {
        "run": run.name, "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "observation_history": "6 frames x 45, newest first; caller maintains history",
        "output_semantics": "raw action; caller clips and applies q_target=q_default+0.25*action",
    })
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 270]
    assert session.get_outputs()[0].shape == [1, 12]

    rng = np.random.RandomState(12)
    cases = [np.zeros((1, 270), dtype=np.float32)]
    stand = np.zeros((1, 45), dtype=np.float32)
    stand[0, 8] = -1.0
    for command in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)]:
        frame = stand.copy()
        frame[0, :3] = np.asarray(command) * [2, 2, 0.25]
        cases.append(np.tile(frame, (1, 6)))
    for scale in [0.1, 1.0, 10.0, 100.0]:
        for _ in range(64):
            cases.append(np.clip(rng.randn(1, 270) * scale, -100, 100).astype(np.float32))
    max_onnx_error = max_jit_error = 0.0
    with torch.no_grad():
        for obs in cases:
            tensor = torch.from_numpy(obs)
            expected = reference.act_inference(tensor).numpy()
            jit_actions = model(tensor).numpy()
            actions = session.run(["actions"], {"obs": obs})[0]
            np.testing.assert_allclose(jit_actions, expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(actions, expected, atol=1e-4, rtol=1e-4)
            max_jit_error = max(max_jit_error, float(np.abs(jit_actions - expected).max()))
            max_onnx_error = max(max_onnx_error, float(np.abs(actions - expected).max()))

    report = {
        "run": run.name, "checkpoint": str(checkpoint), "iteration": saved["iter"],
        "checkpoint_sha256": sha256(checkpoint), "jit": str(jit_path),
        "jit_sha256": sha256(jit_path), "onnx": str(output), "onnx_sha256": sha256(output),
        "torch_version": torch.__version__, "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__, "opset": 11,
        "input": {"name": "obs", "shape": [1, 270], "dtype": "float32"},
        "output": {"name": "actions", "shape": [1, 12], "dtype": "float32"},
        "validation": {"cases": len(cases), "seed": 12,
                       "jit_checkpoint_weights_equal": True,
                       "jit_max_abs_error": max_jit_error,
                       "onnx_max_abs_error": max_onnx_error,
                       "onnx_atol": 1e-4, "onnx_rtol": 1e-4,
                       "onnx_checker": "passed", "result": "passed"},
    }
    report_path = output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
