"""Export a checkpoint-matched Mini Cheetah HIM JIT controller to ONNX on CPU.

Requires torch and onnx; default validation also requires onnxruntime. No Isaac Gym simulation is needed.
The paired model_<iteration>.jit must already exist in exported/policies.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
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
    parser.add_argument("--backend", choices=("onnxruntime", "reference"), default="onnxruntime",
                        help="Validation engine; reference uses ONNX ReferenceEvaluator on CPU")
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
    frame_dim = state['actor.0.weight'].shape[1] - 19
    history_dim = state['estimator.encoder.0.weight'].shape[1]
    critic_dim = state['critic.0.weight'].shape[1]
    if frame_dim not in (45, 47) or history_dim != frame_dim * 6:
        raise ValueError('Expected six 45-D or 47-D Mini Cheetah HIM frames')
    contract = saved.get('observation_contract')
    if frame_dim == 47 and (not contract or contract.get('version') != 'minich_stair_clock_v1'
                            or contract.get('frame_dim') != 47):
        raise ValueError('Clock policy requires checkpoint-matched observation_contract')
    reference = HIMActorCritic(history_dim, critic_dim, frame_dim, 12, activation="elu").cpu().eval()
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
        model, torch.zeros(1, history_dim), str(output), export_params=True,
        opset_version=11, do_constant_folding=True,
        input_names=["obs"], output_names=["actions"],
    )
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(graph, {
        "run": run.name, "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "observation_history": f"6 frames x {frame_dim}, newest first; caller maintains history",
        "observation_contract": json.dumps(contract or {"version": "minich_legacy_45"}),
        "output_semantics": "raw action; caller clips and applies q_target=q_default+0.25*action",
    })
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(output))
    ort_version = None
    if args.backend == "onnxruntime":
        import onnxruntime as ort
        ort_version = ort.__version__
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(output), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        assert session.get_inputs()[0].shape == [1, history_dim]
        assert session.get_outputs()[0].shape == [1, 12]
    else:
        from onnx.reference import ReferenceEvaluator
        session = ReferenceEvaluator(graph)
        assert [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim] == [1, history_dim]
        assert [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] == [1, 12]

    rng = np.random.RandomState(12)
    cases = [np.zeros((1, history_dim), dtype=np.float32)]
    stand = np.zeros((1, frame_dim), dtype=np.float32)
    stand[0, 8] = -1.0
    for command in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)]:
        frame = stand.copy()
        frame[0, :3] = np.asarray(command) * [2, 2, 0.25]
        cases.append(np.tile(frame, (1, 6)))
    if frame_dim == 47:
        for phase in np.linspace(0, 1, 17):
            frames = np.tile(stand, (6, 1))
            phases = phase - np.arange(6) * contract['frequency_hz'] * contract['policy_dt']
            frames[:, 45] = np.sin(2 * np.pi * phases)
            frames[:, 46] = np.cos(2 * np.pi * phases)
            cases.append(frames.reshape(1, history_dim).astype(np.float32))
    for scale in [0.1, 1.0, 10.0, 100.0]:
        for _ in range(64):
            cases.append(np.clip(rng.randn(1, history_dim) * scale, -100, 100).astype(np.float32))
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
        "onnxruntime_version": ort_version, "validation_backend": args.backend, "opset": 11,
        "input": {"name": "obs", "shape": [1, history_dim], "dtype": "float32"},
        "output": {"name": "actions", "shape": [1, 12], "dtype": "float32"},
        "observation_contract": contract,
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
