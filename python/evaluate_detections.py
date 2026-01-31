"""Evaluate pre-generated detections against COCO ground truth."""

import argparse
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_detections(coco_ann_path, detections_json, verbose=False):
    """Evaluate detections against COCO ground truth.
    
    Args:
        coco_ann_path: Path to COCO annotations file
        detections_json: Path to detections JSON file
        verbose: Print detailed output
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading COCO annotations from: {coco_ann_path}")
    coco_gt = COCO(coco_ann_path)
    
    print(f"Loading detections from: {detections_json}")
    with open(detections_json, 'r') as f:
        detections = json.load(f)
    
    if not detections:
        print("❌ No detections found!")
        return {}
    
    print(f"✓ Loaded {len(detections)} predictions")
    
    # Load predictions as COCO results
    print("\nPreparing results...")
    coco_dt = coco_gt.loadRes(detections)
    
    # Run evaluation
    print("\nRunning COCO evaluation...")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Only evaluate on images we have predictions for
    predicted_img_ids = sorted(set(p['image_id'] for p in detections))
    coco_eval.params.imgIds = predicted_img_ids
    
    print(f"Evaluating on {len(predicted_img_ids)} images with predictions")
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    stats = coco_eval.stats
    metrics = {
        'AP': stats[0],
        'AP50': stats[1],
        'AP75': stats[2],
        'AP_small': stats[3],
        'AP_medium': stats[4],
        'AP_large': stats[5],
        'AR1': stats[6],
        'AR10': stats[7],
        'AR100': stats[8],
        'AR_small': stats[9],
        'AR_medium': stats[10],
        'AR_large': stats[11]
    }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Detections file: {detections_json}")
    print(f"Predictions: {len(detections)}")
    print(f"Images with predictions: {len(predicted_img_ids)}")
    print("\nMetrics:")
    for key, val in metrics.items():
        print(f"  {key:15s}: {val:.4f}")
    
    return metrics, coco_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-generated COCO detections.")
    parser.add_argument('--detections', type=str, required=True, 
                       help="Path to detections JSON file")
    parser.add_argument('--coco_ann', type=str, 
                       default='/home/rpi5/projects/coco_data/annotations/instances_val2017.json',
                       help="Path to COCO annotations file")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.detections).exists():
        print(f"❌ Detections file not found: {args.detections}")
        exit(1)
    
    if not Path(args.coco_ann).exists():
        print(f"❌ Annotations file not found: {args.coco_ann}")
        exit(1)
    
    # Evaluate
    metrics, coco_eval = evaluate_detections(
        args.coco_ann,
        args.detections,
        verbose=args.verbose
    )
