#!/home/rpi5/hailo-apps/venv_hailo_apps/bin/python
"""Single image detection with ONNX Runtime using yolo26n.onnx"""

import argparse
import time
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

def letterbox(img, target_size=640, color=(114, 114, 114)):
    """Letterbox resize: preserve aspect ratio, pad to square.
    
    Returns:
        resized_padded: (target_size, target_size, 3) image
        scale: the scale factor applied
        pad_w: horizontal padding (left side)
        pad_h: vertical padding (top side)
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h

def preprocess_image(img_path, target_size=640):
    """Preprocess with letterbox (matching Ultralytics val pipeline)."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    padded, scale, pad_w, pad_h = letterbox(img_rgb, target_size)
    
    # Normalize [0,1] and HWC -> CHW
    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, img, (orig_h, orig_w), scale, pad_w, pad_h

def draw_detections(img, detections, conf_threshold=0.25):
    """
    Draws bounding boxes on the image.
    detections: list of [x1, y1, x2, y2, conf, cls_id]
    """
    h, w = img.shape[:2]
    output_img = img.copy()
    
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det
        
        if conf < conf_threshold:
            continue
            
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = colors[i % len(colors)]
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        
        cls_name = COCO_CLASSES.get(int(cls_id), f"Id {int(cls_id)}")
        label = f"{cls_name} {conf:.2f}"
        
        # Text background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(output_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return output_img

def main():
    parser = argparse.ArgumentParser(description="YOLO26 ONNX Detection")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/yolo26n.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output_onnx.jpg", help="Output path")
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return
        
    # 1. Load Model
    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 2. Preprocess (letterbox)
    print(f"Processing image: {args.image}")
    input_tensor, orig_img, (orig_h, orig_w), scale, pad_w, pad_h = preprocess_image(args.image)
    
    # 3. Inference
    t0 = time.perf_counter()
    outputs = session.run([output_name], {input_name: input_tensor})
    t1 = time.perf_counter()
    print(f"Inference time: {(t1 - t0)*1000:.2f}ms")
    
    # 4. Post-process
    # Output shape: [1, 300, 6] -> [x1, y1, x2, y2, conf, cls] in 640x640 letterboxed space
    raw_detections = outputs[0][0]  # Shape [300, 6]
    
    final_detections = []
    
    for det in raw_detections:
        x1_lb, y1_lb, x2_lb, y2_lb, conf, cls_id = det
        
        if conf < args.conf:
            continue
        
        # Reverse letterbox: subtract padding, then divide by scale
        x1 = (x1_lb - pad_w) / scale
        y1 = (y1_lb - pad_h) / scale
        x2 = (x2_lb - pad_w) / scale
        y2 = (y2_lb - pad_h) / scale
        
        # Clip to original image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
            
        print(f"Det: {COCO_CLS(cls_id)} {conf:.2f} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        final_detections.append([x1, y1, x2, y2, conf, cls_id])
        
    print(f"Found {len(final_detections)} detections.")
    
    # 5. Draw
    out_img = draw_detections(orig_img, final_detections, args.conf)
    cv2.imwrite(args.output, out_img)
    print(f"Saved to {args.output}")

def COCO_CLS(cls_id):
    """Quick helper for print"""
    names = {0: 'person', 56: 'chair', 62: 'tv', 72: 'refrigerator', 75: 'vase', 58: 'potted plant', 74: 'clock'}
    return names.get(int(cls_id), f"cls{int(cls_id)}")

if __name__ == "__main__":
    main()
