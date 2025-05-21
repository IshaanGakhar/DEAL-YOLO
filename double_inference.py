import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json
import time
import logging
import concurrent.futures
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("double_inference.log"),
        logging.StreamHandler()
    ]
)

IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45

def calculate_iou(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
        
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    if area_box1 <= 0 or area_box2 <= 0:
        return 0.0
    
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union if area_union > 0 else 0.0
    
    return iou

def calculate_optimal_crop(detection, img_width, img_height, pad_factor=0.2):
    x1, y1, x2, y2 = detection['bbox']
    sw = max(1, x2 - x1)  
    sh = max(1, y2 - y1)  
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  
    
    aspect_ratio = sw / sh if sh > 0 else 1  
    
    pad_w = sw * pad_factor
    pad_h = sh * pad_factor
    
    crop_w = sw + 2 * pad_w
    crop_h = sh + 2 * pad_h
    
    new_x1 = max(0, int(cx - crop_w / 2))
    new_y1 = max(0, int(cy - crop_h / 2))
    new_x2 = min(img_width, int(cx + crop_w / 2))
    new_y2 = min(img_height, int(cy + crop_h / 2))
    
    actual_w = new_x2 - new_x1
    actual_h = new_y2 - new_y1

    if actual_w < 10 or actual_h < 10:
        min_size = 32
        new_x1 = max(0, int(cx - min_size / 2))
        new_y1 = max(0, int(cy - min_size / 2))
        new_x2 = min(img_width, int(cx + min_size / 2))
        new_y2 = min(img_height, int(cy + min_size / 2))
    
    return {
        'x1': new_x1,
        'y1': new_y1,
        'x2': new_x2,
        'y2': new_y2
    }

def prepare_cropped_image(img, crop_info):

    crop = img.crop((crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']))

    original_w, original_h = crop.size
    ratio = min(640 / original_w, 640 / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    resized = crop.resize(new_size, Image.BILINEAR)
    
    padded_img = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x = (640 - new_size[0]) // 2
    pad_y = (640 - new_size[1]) // 2
    padded_img.paste(resized, (pad_x, pad_y))
    
    return {
        'image': padded_img,
        'original_size': (original_w, original_h),
        'new_size': new_size,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'ratio': ratio
    }

def scale_boxes(boxes, pad_x, pad_y, crop_info, ratio):

    if boxes.size == 0:
        return np.array([])
    
    scaled = boxes.copy()
    
    scaled[:, 0] -= pad_x
    scaled[:, 1] -= pad_y
    scaled[:, 2] -= pad_x
    scaled[:, 3] -= pad_y
    
    scaled[:, 0] /= ratio
    scaled[:, 1] /= ratio
    scaled[:, 2] /= ratio
    scaled[:, 3] /= ratio
    
    scaled[:, 0] += crop_info['x1']
    scaled[:, 1] += crop_info['y1']
    scaled[:, 2] += crop_info['x1']
    scaled[:, 3] += crop_info['y1']
    
    return scaled

def non_max_suppression(boxes, scores, labels, iou_threshold=0.45):

    if not boxes or len(boxes) == 0:
        return [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        area_intersection = w * h
        
        area_box1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_boxes = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        
        union = area_box1 + area_boxes - area_intersection
        iou = area_intersection / union
        
        same_label_mask = labels[idxs[1:]] == labels[i]
        overlap_mask = iou > iou_threshold
        remove_mask = same_label_mask & overlap_mask
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(remove_mask)[0] + 1)))
    
    return boxes[pick].tolist(), scores[pick].tolist(), labels[pick].tolist()

def perform_double_inference(image_path, model, original_detection, use_augment=False):
    try:

        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        
        crop_info = calculate_optimal_crop(original_detection, img_width, img_height)
        
        if crop_info['x2'] <= crop_info['x1'] or crop_info['y2'] <= crop_info['y1']:
            logging.warning(f"Invalid crop dimensions: {crop_info}")
            return None
        processed = prepare_cropped_image(img, crop_info)
        with torch.no_grad():
            new_results = model.predict(processed['image'], verbose=False, augment=use_augment)
        if len(new_results[0].boxes) == 0:
            return None

        boxes = new_results[0].boxes.xyxy.cpu().numpy()
        confs = new_results[0].boxes.conf.cpu().numpy()
        labels = new_results[0].boxes.cls.cpu().numpy().astype(int)

        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
            confs = np.expand_dims(confs, axis=0)
            labels = np.expand_dims(labels, axis=0)

        scaled_boxes = scale_boxes(
            boxes,
            processed['pad_x'],
            processed['pad_y'],
            crop_info,
            processed['ratio']
        )

        return process_refined_boxes(scaled_boxes, labels, confs, original_detection, img_width, img_height)
    
    except Exception as e:
        logging.error(f"Error in double inference: {str(e)}")
        return None

def process_refined_boxes(scaled_boxes, labels, confs, original_detection, img_width, img_height):

    if scaled_boxes.size == 0 or len(labels) == 0 or len(confs) == 0:
        return None

    if not (len(scaled_boxes) == len(labels) == len(confs)):
        logging.warning(f"Mismatched array lengths: boxes={len(scaled_boxes)}, labels={len(labels)}, confs={len(confs)}")
        return None
    
    original_label = original_detection['category_id']
    original_score = original_detection['score']

    best_match = None
    best_combined = -1
    best_conf = -1
    
    for i, (box, label, conf) in enumerate(zip(scaled_boxes, labels, confs)):
        if label != original_label:
            continue

        if (box[2] <= box[0]) or (box[3] <= box[1]) or \
           (box[0] < 0) or (box[1] < 0) or \
           (box[2] > img_width) or (box[3] > img_height):
            continue

        current_iou = calculate_iou(original_detection['bbox'], box)
        
        if current_iou < 0.25:
            continue

        combined_score = (conf * 0.6) + (current_iou * 0.4)
        
        if combined_score > best_combined:
            best_combined = combined_score
            best_conf = conf
            best_match = {
                'bbox': box.tolist(),
                'score': float(conf),
                'category_id': int(label)
            }
    return best_match if best_match and best_conf > original_score else None

def calculate_metrics(predictions, targets):

    if not predictions or not targets:
        return {'map_50': 0, 'precision': 0, 'recall': 0}
    
    metric = MeanAveragePrecision(class_metrics=True, max_detection_thresholds=None)
    metric.update(predictions, targets)
    results = metric.compute()
    
    tp, fp, fn = 0, 0, 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].numpy()
        pred_scores = pred['scores'].numpy()
        pred_labels = pred['labels'].numpy()
        
        target_boxes = target['boxes'].numpy()
        target_labels = target['labels'].numpy()
        
        matched_targets = set()
        
        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_idx = -1
            
            for j, (gt_box, gt_label) in enumerate(zip(target_boxes, target_labels)):
                if j in matched_targets:
                    continue
                    
                if label == gt_label:
                    iou = calculate_iou(box, gt_box)
                    if iou > best_iou and iou >= IOU_THRESHOLD:
                        best_iou = iou
                        best_idx = j
            
            if best_idx >= 0:
                tp += 1
                matched_targets.add(best_idx)
            else:
                fp += 1
        
        fn += len(target_boxes) - len(matched_targets)
    
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    
    return {
        'map_50': results['map_50'].item(),
        'precision': precision,
        'recall': recall,
        'class_metrics': {
            f'class_{i}': {
                'ap': ap.item()
            } for i, ap in enumerate(results['map_per_class'])
        }
    }

def load_image_predictions(predictions_path, conf_threshold=0.25):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"❌ Predictions file not found at {predictions_path}")
    
    with open(predictions_path, "r") as f:
        val_predictions = json.load(f)
    
    image_predictions = {}
    for pred in val_predictions:
        image_name = pred["image_id"]
        if image_name not in image_predictions:
            image_predictions[image_name] = {"boxes": [], "scores": [], "labels": []}
        
        if pred["score"] >= conf_threshold:
            x, y, w, h = pred["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            image_predictions[image_name]["boxes"].append([x1, y1, x2, y2])
            image_predictions[image_name]["scores"].append(pred["score"])
            image_predictions[image_name]["labels"].append(pred["category_id"])
    
    return image_predictions

def load_ground_truth(label_path, img_width, img_height):
    boxes, labels = [], []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id))
    
    return boxes, labels

def process_image(image_path, model, image_predictions, model_names, use_augment=False):
    try:
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if image_name not in image_predictions:
            logging.warning(f"No predictions found for {image_name}")
            return None
        
        label_path = os.path.join(LABEL_DIR, f"{image_name}.txt")
        true_boxes, true_labels = load_ground_truth(label_path, img_width, img_height)
        
        current_predictions = image_predictions[image_name]
        
        initial_preds = {
            'boxes': current_predictions['boxes'].copy(),
            'scores': current_predictions['scores'].copy(),
            'labels': current_predictions['labels'].copy()
        }

        refined_candidates = []
        
        for idx in range(len(current_predictions['scores'])):
            if current_predictions['scores'][idx] >= CONF_THRESHOLD:
                original_detection = {
                    'bbox': current_predictions['boxes'][idx],
                    'score': current_predictions['scores'][idx],
                    'category_id': current_predictions['labels'][idx]
                }
                
                try:
                    refined = perform_double_inference(
                        image_path,
                        model,
                        original_detection,
                        use_augment
                    )
                    
                    if refined is not None:
                        refined_candidates.append({
                            'idx': idx,
                            'bbox': refined['bbox'],
                            'score': refined['score'],
                            'label': refined['category_id']
                        })
                except Exception as e:
                    logging.error(f"Error refining detection {idx} in {image_name}: {str(e)}")
        
        for candidate in refined_candidates:
            i = candidate['idx']
            current_predictions['boxes'][i] = candidate['bbox']
            current_predictions['scores'][i] = candidate['score']
            current_predictions['labels'][i] = candidate['label']
        
        if current_predictions['boxes'] and current_predictions['scores'] and current_predictions['labels']:
            current_predictions['boxes'], current_predictions['scores'], current_predictions['labels'] = \
                non_max_suppression(
                    current_predictions['boxes'],
                    current_predictions['scores'],
                    current_predictions['labels'],
                    NMS_IOU_THRESHOLD
                )
        
        results = {
            'image_path': image_path,
            'refined_predictions': {
                'boxes': current_predictions['boxes'],
                'scores': current_predictions['scores'],
                'labels': current_predictions['labels']
            },
            'ground_truth': {
                'boxes': true_boxes,
                'labels': true_labels
            }
        }
        
        return results
    
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    """Main execution function"""
    start_time = time.time()
    
    logging.info("Loading YOLO model...")
    model = YOLO(MODEL_WEIGHTS)
    model_names = model.names

    logging.info("Loading predictions...")
    predictions_path = "/kaggle/input/json-files/spdp2p2.json"
    image_predictions = load_image_predictions(predictions_path, CONF_THRESHOLD)
    
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                  if os.path.splitext(f)[0] in image_predictions]
    
    logging.info(f"Processing {len(image_files)} images with double inference...")
    
    results = {}
    all_predictions = []
    all_targets = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_image = {
            executor.submit(
                process_image,
                image_path,
                model,
                image_predictions,
                model_names,
                True  
            ): image_path for image_path in image_files
        }
        
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                if result:
                    results[image_path] = result
                    if result['refined_predictions']['boxes'] and result['ground_truth']['boxes']:
                        pred = {
                            'boxes': torch.tensor(result['refined_predictions']['boxes']),
                            'scores': torch.tensor(result['refined_predictions']['scores']),
                            'labels': torch.tensor(result['refined_predictions']['labels']),
                        }
                        target = {
                            'boxes': torch.tensor(result['ground_truth']['boxes']),
                            'labels': torch.tensor(result['ground_truth']['labels']),
                        }
                        all_predictions.append(pred)
                        all_targets.append(target)
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")

    if not all_predictions or not all_targets:
        logging.warning("No valid predictions or targets collected, cannot calculate metrics")
        return
    
    if len(all_predictions) < 1:  
        logging.warning(f"Only {len(all_predictions)} prediction sets collected, metrics may not be reliable")

    metrics = calculate_metrics(all_predictions, all_targets)
    
    logging.info(f"Results after double inference:")
    logging.info(f"mAP@0.5: {metrics['map_50']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    for class_id, class_metrics in metrics['class_metrics'].items():
        logging.info(f"{class_id} AP: {class_metrics['ap']:.4f}")
    print(f"Results after double inference:")
    print(f"mAP@0.5: {metrics['map_50']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    
    total_time = time.time() - start_time
    num_images = len(results)
    total_detections = sum(len(data['refined_predictions']['boxes']) for data in results.values())
    
    logging.info(f"Processed {num_images} images with {total_detections} detections in {total_time:.2f} seconds")
    logging.info(f"Average time per image: {total_time/max(1, num_images):.2f} seconds")

if __name__ == "__main__":
    main()