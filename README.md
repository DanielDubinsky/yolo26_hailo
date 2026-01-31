# YOLO26 Object Detection on Raspberry Pi 5 + Hailo-8L

This repository provides a complete pipeline for deploying YOLO26n object detection models on the Raspberry Pi 5 AI Kit (Hailo-8L NPU). It includes scripts for model conversion (ONNX → HEF), C++ inference/evaluation code, and Python inference examples.

## repository Structure

```
.
├── cpp/                 # C++ inference and COCO evaluation code
├── export/              # Scripts to convert ONNX to Hailo HEF format
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

The export process involves 3 steps: separating the model into backbone/head (automatically handled by the HAR parser), quantizing the backbone, and compiling to HEF.

> [!NOTE]
> These export scripts (`export/*.py`) require the **Hailo Dataflow Compiler (DFC)** environment.

**Files required:** `models/yolo26n.onnx`.

### Step 1: Parse ONNX to HAR

Extracts the backbone from the ONNX model.

```bash
# From the root directory
python export/1_onnx_to_har.py models/yolo26n.onnx models/yolo26n.har
```

### Step 2: Quantize HAR

Quantizes the model using calibration images. You need a directory with ~1024 calibration images (e.g., from COCO train/val) in `data/calib_images`.

```bash
# Ensure you have calibration images in data/calib_images
python export/2_quantize_har.py models/yolo26n.har models/yolo26n_quantized.har data/calib_images
```

### Step 3: Compile to HEF

Compiles the quantized model to a Hailo Executable Format (HEF) binary.

```bash
python export/3_compile_hef.py models/yolo26n_quantized.har models/yolo26n.hef
```

---

## 3. Python Inference

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

## 4. C++ Inference & Evaluation

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
