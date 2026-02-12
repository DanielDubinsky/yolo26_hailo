#!/home/rpi5/hailo-apps/venv_hailo_apps/bin/python
import numpy as np
import argparse
import time
import json
from pathlib import Path
import onnxruntime as ort
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def letterbox(img, target_size=640, color=(114, 114, 114)):
    """Letterbox resize: preserve aspect ratio, pad to square.
    
    Returns:
        resized_padded: (target_size, target_size, 3) image
        scale: the scale factor applied
        pad_w: horizontal padding (left side)
        pad_h: vertical padding (top side)
    """
    h, w = img.shape[:2]
    
    # Scale to fit within target_size
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # Calculate padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Pad to target_size x target_size
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h

def preprocess_image(img_path, target_size=640):
    """Preprocess with letterbox (matching Ultralytics val pipeline)."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    padded, scale, pad_w, pad_h = letterbox(img_rgb, target_size)
    
    # Normalize to [0, 1] and transpose from HWC to CHW
    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, (orig_h, orig_w), scale, pad_w, pad_h

def run_onnx_evaluation(args):
    """Run ONNX model evaluation on COCO dataset"""
    print(f"[INFO] Loading ONNX model: {args.onnx_model}")
    session = ort.InferenceSession(args.onnx_model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"[INFO] Loading COCO annotations from: {args.coco_ann}")
    coco_gt = COCO(args.coco_ann)
    image_ids = coco_gt.getImgIds()
    
    if args.limit > 0:
        image_ids = image_ids[:args.limit]
    
    # Mapping from Model Index (0-79) to COCO Category ID
    COCO_IDS_MAP = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    
    detections = []
    
    print(f"[INFO] Running evaluation on {len(image_ids)} images...")
    for i, image_id in enumerate(image_ids):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_ids)} images")
            
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = str(Path(args.coco_images_dir) / image_info['file_name'])
        
        try:
            input_tensor, (orig_h, orig_w), scale, pad_w, pad_h = preprocess_image(image_path)
        except ValueError as e:
            print(f"Skipping image {image_path}: {e}")
            continue
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})[0]
        
        # Post-process: output is [1, 300, 6] -> [x1, y1, x2, y2, conf, cls]
        # Coordinates are in the 640x640 letterboxed space
        raw_detections = outputs[0]  # [300, 6]
        
        for det in raw_detections:
            x1_lb, y1_lb, x2_lb, y2_lb, conf, cls_id = det
            
            if conf >= args.conf_threshold:
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
                
                # Convert [x1, y1, x2, y2] to [x_min, y_min, width, height] for COCO
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    continue
                
                # Category mapping
                category_id = int(cls_id)
                if 0 <= category_id < len(COCO_IDS_MAP):
                    real_category_id = COCO_IDS_MAP[category_id]
                else:
                    real_category_id = category_id

                detections.append({
                    'image_id': image_id,
                    'category_id': real_category_id,
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(conf),
                })

    # Save detections to a temporary file
    detections_file = f'detections_{Path(args.onnx_model).stem}.json'
    with open(detections_file, 'w') as f:
        json.dump(detections, f)
    
    print(f"\n[INFO] Detections saved to {detections_file}")
    
    if len(detections) == 0:
        print("[WARNING] No detections found.")
        return

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(detections_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Only evaluate on the images we actually processed
    coco_eval.params.imgIds = image_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Model Evaluation on COCO")
    parser.add_argument("onnx_model", type=str, help="Path to ONNX model")
    parser.add_argument("coco_images_dir", type=str, help="Path to COCO validation images directory")
    parser.add_argument("--coco_ann", type=str, default='data/coco/annotations/instances_val2017.json', help="Path to COCO annotations file")
    parser.add_argument("--conf_threshold", type=float, default=0.001, help="Confidence threshold for detections")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images for evaluation (0 for all)")

    args = parser.parse_args()
    run_onnx_evaluation(args)
