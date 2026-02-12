# YOLO26 Object Detection on Raspberry Pi 5 + Hailo-8L

This repository provides a complete pipeline for deploying YOLO26n object detection models on the Raspberry Pi 5 AI Kit (Hailo-8L NPU). It includes scripts for model conversion (ONNX → HEF), C++ inference/evaluation code, and Python inference examples.

## Performance Summary (Raspberry Pi 5 + Hailo-8L)

| Model | CPU mAP (FP32) | CPU FPS | Hailo mAP (INT8) | Hailo FPS | Speedup | Accuracy Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 0.402 | 6.50 | 0.371 | 86.5 | 13.3x | 92.3% |
| **YOLO26s** | 0.477 | 2.62 | 0.424 | 37.5 | 14.3x | 88.9% |
| **YOLO26m** | 0.525 | 0.88 | 0.441 | 23.4 | 26.6x | 84.0% |
| **YOLO26l** | 0.541 | 0.74 | 0.473 | 17.9 | 24.2x | 87.4% |

*\*Tested on COCO val2017 with letterbox preprocessing. FPS measured end-to-end (preprocessing + inference + postprocessing).*

## repository Structure

```
.
├── cpp/                 # C++ inference and COCO evaluation code
├── export/              # Python package for model conversion (ONNX → HEF)
├── models/              # Place your .onnx and .hef models here
├── python/              # Python inference scripts
├── data/                # Data directory (calibration images, COCO val)
├── requirements.txt     # Python dependencies
├── setup.py             # Setup script
├── LICENSE              # MIT License
└── README.md            # This file
```

## Prerequisites

1.  **Hardware**: Raspberry Pi 5 with Hailo-8L AI Kit.
2.  **Software**: 
    - Raspberry Pi OS (64-bit)
    - [HailoRT](https://github.com/hailo-ai/hailort) installed.
    - [Hailo Dataflow Compiler (DFC)](https://github.com/hailo-ai/hailo_dataflow_compiler) (required for export scripts).

## Environment Setup

This project assumes you are running inside a Python virtual environment that has `hailo-platform` (HailoRT Python API) installed. If you are using the official Hailo examples environment:

```bash
source ~/hailo-apps/venv_hailo_apps/bin/activate
```

To install additional dependencies for this project:

```bash
pip install -r requirements.txt
```


## Data Setup

To run quantization or COCO evaluation, you need to download the datasets.

### Calibration Data (for Quantization)

Download 1024 random images from COCO Train 2017 to `data/calib_images`:

```bash
# Requires fiftyone
pip install fiftyone
python scripts/download_calib.py
```

### COCO Validation Data (for Evaluation)

Download standard COCO Val 2017 images and annotations to `data/coco`:

```bash
# Downloads ~1GB
bash scripts/download_coco.sh
```



## 1. Model Preparation

### Option A: Download Pre-compiled HEF (Recommended)

If you just want to run inference, download the pre-compiled Hailo binary from the [Releases](https://github.com/DanielDubinsky/yolo26_hailo/releases) page.

```bash
bash scripts/download_hef.sh
```

### Option B: Export from ONNX

If you want to compile the model yourself:

1. Obtain the YOLO26n ONNX model:
```bash
# Downloads YOLO26n and exports to models/yolo26n.onnx
pip install ultralytics
python scripts/download_model.py
```

Optional: If you wish to inspect or run the separated backbone/head on CPU/ONNX Runtime manually, you can split the ONNX model:

```bash
python export/0_extract_subgraphs.py models/yolo26n.onnx models/
```

---

## 2. Model Export (ONNX → HEF)

The export process is fully automated using `export.cli`. This package handles extracting subgraphs, parsing to HAR, quantizing, and compiling to HEF. It creates a unique experiment directory for each run with full logs and artifacts.

> [!NOTE]
> These export scripts require the **Hailo Dataflow Compiler (DFC)** environment.

### Usage

```bash
# Run from the repository root
python -m export.cli \
  --variant yolo26n \
  --target hailo8l \
  --onnx models/yolo26n.onnx \
  --calib_dir data/coco/val2017 \
  --tag my_experiment
```

### Arguments
-   `--variant`: `yolo26n` (default), `yolo26s`, `yolo26m`, `yolo26l`.
-   `--target`: `hailo8l` (default), `hailo8`.
-   `--onnx`: Path to the input ONNX model.
-   `--calib_dir`: Directory containing calibration images.
-   `--alls`: (Optional) Path to a custom `.alls` model script.
-   `--tag`: (Optional) Custom tag for the experiment run name.

### Configuration
-   **`export/config.py`**: Pydantic-based configuration and variant definitions.
-   **Output**: Results are saved in `experiments/{VARIANT}_{TARGET}_{TIMESTAMP}/`.
    -   `artifacts/3_compiled/model.hef`: Final compiled binary.
    -   `run.log`: Full execution log.
    -   `model_script.alls`: The model script used (if any).

---

## 3. ONNX Verification Tools

The repository includes tools to run the original ONNX models on CPU for verification and debugging. These are located in `cpp/onnx/` and `python/onnx/`.

### Python ONNX Inference

```bash
# Single Image Detection
python python/onnx/detect_image.py input.jpg --model models/yolo26n.onnx

# Benchmark
python python/onnx/benchmark.py models/yolo26n.onnx --iterations 100
```

### C++ ONNX Inference

To build the ONNX C++ tools:

```bash
cd cpp/onnx
make
cd ../..
```

Then run:

```bash
# Detect Image
./cpp/onnx/detect_image input.jpg models/yolo26n.onnx

# Benchmark
./cpp/onnx/benchmark models/yolo26n.onnx 100
```

---

## 4. Python Inference (Hailo)

Run inference using the generated HEF file and Python post-processing.

### Single Image Detection

```bash
# Run detection on an image
python python/detect_image.py input.jpg --hef models/yolo26n.hef --output output.jpg
```

Arguments:
- `--hef`: Path to the .hef file (default: `../models/yolo26n.hef`)
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--normalize`: Use if your model expects [0,1] normalized float input (usually False for Hailo uint8 models).

### Benchmark

Measure performance (FPS) without I/O overhead.

```bash
python python/benchmark_inference.py --hef models/yolo26n.hef --iterations 1000
```

---

## 5. C++ Inference & Evaluation

The C++ implementation provides high-performance inference.

### Build

```bash
cd cpp
make
# Generates: detect_image, benchmark_inference, run_coco_inference
cd ..
```

### Run Benchmark

```bash
./cpp/benchmark_inference models/yolo26n.hef 1000
```

### Run Detection on Image

```bash
./cpp/detect_image input.jpg models/yolo26n.hef
```

### Evaluate on COCO (mAP)

To reproduce accuracy results, run the evaluation tool on the COCO validation set.

1.  Download COCO validation images to `data/coco/val2017`.
2.  Run inference to generate results JSON:

```bash
./cpp/run_coco_inference data/coco/val2017 models/yolo26n.hef detections.json
```

3.  Calculate mAP using the provided Python script:

```bash
python python/evaluate_detections.py --detections detections.json --coco_ann data/coco/annotations/instances_val2017.json
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
