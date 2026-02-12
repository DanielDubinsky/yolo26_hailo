#!/home/rpi5/hailo-apps/venv_hailo_apps/bin/python
import numpy as np
import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import onnxruntime as ort

@dataclass
class OnnxInferenceStats:
    """Runtime statistics for ONNX inference"""
    preprocess_time: float
    inference_time: float
    postprocess_time: float
    total_time: float

def run_onnx_benchmark(args):
    """Run ONNX inference with statistics collection"""
    print(f"[INFO] Loading ONNX model: {args.onnx_model}")
    try:
        session = ort.InferenceSession(args.onnx_model)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    input_name = session.get_inputs()[0].name
    # Handle multiple outputs if present, though we might only care about the first one for benchmarking
    output_names = [o.name for o in session.get_outputs()]
    
    print(f"[INFO] Input Name: {input_name}")
    print(f"[INFO] Output Names: {output_names}")

    stats_list = []

    # Warmup
    print("[INFO] Warming up the model...")
    input_shape = session.get_inputs()[0].shape
    # Handle dynamic batch size if needed, default to 1 if 'batch' or unk
    batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
    h, w = input_shape[2], input_shape[3]
    
    for _ in range(5):
        warmup_input = np.random.rand(batch_size, 3, h, w).astype(np.float32)
        session.run(output_names, {input_name: warmup_input})

    print(f"[INFO] Running benchmark for {args.iterations} iterations...")
    for i in range(args.iterations):
        t_start_total = time.perf_counter()

        # Preprocess
        t_pre = time.perf_counter()
        # Generate random data, normalized float32 [0,1]
        img_data = np.random.rand(h, w, 3).astype(np.float32)
        # Transpose from (H, W, C) to (C, H, W)
        input_data = np.transpose(img_data, (2, 0, 1))
        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        preprocess_time = time.perf_counter() - t_pre

        # Infer
        t_inf = time.perf_counter()
        session.run(output_names, {input_name: input_data})
        inference_time = time.perf_counter() - t_inf

        # Postprocess (dummy)
        t_post = time.perf_counter()
        # In a real scenario, you'd process the output here
        postprocess_time = time.perf_counter() - t_post

        total_time = time.perf_counter() - t_start_total

        stats_list.append(OnnxInferenceStats(
            preprocess_time=preprocess_time,
            inference_time=inference_time,
            postprocess_time=postprocess_time,
            total_time=total_time
        ))

    # Print summary
    print("\n" + "="*60)
    print("ONNX INFERENCE BENCHMARK SUMMARY")
    print("="*60)

    inference_times = np.array([s.inference_time for s in stats_list])
    avg_inference = inference_times.mean()
    std_inference = inference_times.std()

    total_times = np.array([s.total_time for s in stats_list])
    avg_total = total_times.mean()
    std_total = total_times.std()
    
    if avg_total > 0:
        fps = 1.0 / avg_total
    else:
        fps = 0.0

    print(f"ONNX Model: {args.onnx_model}")
    print(f"Total Iterations: {args.iterations}")
    print(f"Avg Inference Time: {avg_inference*1000:.2f}ms ± {std_inference*1000:.2f}ms")
    print(f"Avg Total Time (incl. pre/post): {avg_total*1000:.2f}ms ± {std_total*1000:.2f}ms")
    print(f"Throughput: {fps:.2f} FPS")

    # Save statistics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([asdict(s) for s in stats_list], f, indent=2)
        print(f"\n✓ Statistics saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Inference Benchmark")
    parser.add_argument("onnx_model", type=str, help="Path to ONNX model")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    parser.add_argument("--output", type=str, help="JSON file to save statistics")

    args = parser.parse_args()
    run_onnx_benchmark(args)
