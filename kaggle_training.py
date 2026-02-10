"""
Helmet Compliance Detection - Training Script
==============================================
YOLOv8s model for industrial safety monitoring.

Datasets:
1. Hard Hat Detection (andrewmvd) - ~5k images, XML format
2. COCO Negative Mining - Hard negatives (person→head→no_helmet)

Instructions:
1. Create Kaggle notebook with GPU T4 x2
2. Add datasets: andrewmvd/hard-hat-detection, coco-2017-dataset
3. Copy cells and run

Author: Raghavendra
"""

# =============================================================================
# CELL 1: Environment Setup  
# RUN THIS CELL FIRST!
# =============================================================================

# STEP 1: Upgrade ultralytics (run in separate cell, then restart kernel)
# !pip install ultralytics==8.3.50 --quiet

# STEP 2: Apply PyTorch patch BEFORE any ultralytics import
import torch

# Store original function
_original_load = torch.load

# Create patched version
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

# Apply patch globally
torch.load = _safe_load

# Also set the environment variable for any subprocesses
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

# Now import everything else
import shutil
import json
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from tqdm import tqdm
from collections import defaultdict
import random

print("✅ PyTorch 2.6+ patch applied")
print(f"   torch.load patched: {torch.load != _original_load}")

# =============================================================================
# CELL 2: Configuration
# =============================================================================

# Dataset paths
HARDHAT_DIR = Path("/kaggle/input/hard-hat-detection")
COCO_DIR = Path("/kaggle/input/coco-2017-dataset/coco2017")

# Output
WORK_DIR = Path("/kaggle/working")
FINAL_DATASET = WORK_DIR / "helmet_yolo"
TRAIN_DIR = FINAL_DATASET / "train"
VAL_DIR = FINAL_DATASET / "valid"

# Classes
FINAL_CLASSES = {0: "helmet", 1: "no_helmet"}
HARDHAT_MAP = {"helmet": 0, "head": 1, "person": None}

# Filters
MIN_BOX_AREA = 0.001
MAX_BOX_AREA = 0.5
MIN_PERSON_HEIGHT = 80

# Ratios
COCO_RATIO = 0.15  # 15% COCO negatives in training

print("📁 Paths configured")
print(f"   Hard Hat: {HARDHAT_DIR}")
print(f"   COCO: {COCO_DIR}")

# =============================================================================
# CELL 3: Explore Datasets
# =============================================================================

def explore():
    print("\n" + "="*50)
    print("📊 DATASET EXPLORATION")
    print("="*50)
    
    info = {}
    
    # Hard Hat
    print("\n📦 Hard Hat Detection:")
    if HARDHAT_DIR.exists():
        imgs_dir = HARDHAT_DIR / "images"
        annot_dir = HARDHAT_DIR / "annotations"
        if not imgs_dir.exists():
            imgs_dir = annot_dir = HARDHAT_DIR
        
        xmls = list(annot_dir.glob("*.xml"))
        info["hardhat"] = {"images_dir": imgs_dir, "xmls": xmls}
        print(f"   {len(xmls)} XML annotations")
        
        # Sample classes
        if xmls:
            classes = set()
            for x in xmls[:100]:
                tree = ET.parse(x)
                for obj in tree.findall('.//object'):
                    classes.add(obj.find('name').text.lower())
            print(f"   Classes: {classes}")
    
    # COCO
    print("\n📦 COCO 2017:")
    if COCO_DIR.exists():
        train_dir = COCO_DIR / "train2017"
        annot = COCO_DIR / "annotations" / "instances_train2017.json"
        if not annot.exists():
            annot = COCO_DIR / "annotations" / "instances_val2017.json"
            train_dir = COCO_DIR / "val2017"
        
        info["coco"] = {"images_dir": train_dir, "annotations": annot}
        print(f"   Annotations: {annot.name}")
    
    return info

dataset_info = explore()

# =============================================================================
# CELL 4: Parse Hard Hat XML
# =============================================================================

def parse_xml(xml_path, images_dir):
    """Parse Hard Hat XML to YOLO format."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        w, h = int(size.find('width').text), int(size.find('height').text)
        filename = root.find('filename').text
        
        img_path = images_dir / filename
        if not img_path.exists():
            for ext in [".jpg", ".png"]:
                test = images_dir / (xml_path.stem + ext)
                if test.exists():
                    img_path = test
                    break
        
        if not img_path.exists():
            return None
        
        objects = []
        for obj in root.findall('object'):
            cls = obj.find('name').text.lower()
            if cls not in HARDHAT_MAP or HARDHAT_MAP[cls] is None:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            area = ((xmax-xmin)*(ymax-ymin)) / (w*h)
            if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
                continue
            
            x_c = ((xmin+xmax)/2) / w
            y_c = ((ymin+ymax)/2) / h
            bw = (xmax-xmin) / w
            bh = (ymax-ymin) / h
            
            objects.append({"class_id": HARDHAT_MAP[cls], "bbox": [x_c, y_c, bw, bh]})
        
        return {"image_path": img_path, "objects": objects, "source": "hardhat"} if objects else None
    except:
        return None

# =============================================================================
# CELL 5: Collect Hard Hat Samples
# =============================================================================

def collect_hardhat():
    print("\n" + "="*50)
    print("📦 COLLECTING HARD HAT SAMPLES")
    print("="*50)
    
    samples = []
    counts = {0: 0, 1: 0}
    
    if "hardhat" not in dataset_info:
        print("❌ Hard Hat dataset not found!")
        return samples, counts
    
    hh = dataset_info["hardhat"]
    for xml in tqdm(hh["xmls"], desc="Parsing XMLs"):
        s = parse_xml(xml, hh["images_dir"])
        if s:
            samples.append(s)
            for o in s["objects"]:
                counts[o["class_id"]] += 1
    
    print(f"\n✅ Collected {len(samples)} images")
    print(f"   helmet: {counts[0]}, no_helmet: {counts[1]}")
    
    return samples, counts

hardhat_samples, hardhat_counts = collect_hardhat()

# =============================================================================
# CELL 6: COCO Negative Mining
# =============================================================================

def extract_coco_negatives(max_samples=1500):
    """Extract head-level no_helmet from COCO persons."""
    print("\n" + "="*50)
    print("🔍 COCO NEGATIVE MINING")
    print("="*50)
    
    if "coco" not in dataset_info or not dataset_info["coco"]["annotations"]:
        print("⚠️ COCO not available")
        return []
    
    coco_info = dataset_info["coco"]
    
    with open(coco_info["annotations"]) as f:
        coco = json.load(f)
    
    img_lookup = {i["id"]: i for i in coco["images"]}
    person_id = next((c["id"] for c in coco["categories"] if c["name"] == "person"), None)
    
    if not person_id:
        return []
    
    # Group valid persons by image
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == person_id and not a.get("iscrowd"):
            if a["bbox"][3] >= MIN_PERSON_HEIGHT:
                by_image[a["image_id"]].append(a)
    
    print(f"   Valid images: {len(by_image)}")
    
    # Sample
    random.seed(42)
    selected = random.sample(list(by_image.keys()), min(max_samples, len(by_image)))
    
    samples = []
    images_dir = coco_info["images_dir"]
    
    for img_id in tqdm(selected, desc="Processing COCO"):
        info = img_lookup.get(img_id)
        if not info:
            continue
        
        img_path = images_dir / info["file_name"]
        if not img_path.exists():
            continue
        
        w, h = info["width"], info["height"]
        
        objects = []
        for a in by_image[img_id]:
            x, y, bw, bh = a["bbox"]
            
            # Head = top 28%
            head_h = bh * 0.28
            if head_h < 15:
                continue
            
            x_c = (x + bw/2) / w
            y_c = (y + head_h/2) / h
            norm_w = bw / w
            norm_h = head_h / h
            
            objects.append({"class_id": 1, "bbox": [x_c, y_c, norm_w, norm_h]})
        
        if objects:
            samples.append({"image_path": img_path, "objects": objects, "source": "coco"})
    
    print(f"✅ Extracted {len(samples)} COCO negatives")
    return samples

coco_samples = extract_coco_negatives()

# =============================================================================
# CELL 7: Combine & Split
# =============================================================================

def combine_datasets(hardhat, coco):
    print("\n" + "="*50)
    print("⚖️ COMBINING DATASETS")
    print("="*50)
    
    random.seed(42)
    random.shuffle(hardhat)
    
    # 80/20 split
    split = int(len(hardhat) * 0.8)
    train_h = hardhat[:split]
    val = hardhat[split:]
    
    # Add COCO to training only
    max_coco = int(len(train_h) * COCO_RATIO / (1 - COCO_RATIO))
    train_coco = coco[:max_coco]
    
    train = train_h + train_coco
    random.shuffle(train)
    
    print(f"Train: {len(train)} ({len(train_h)} hardhat + {len(train_coco)} COCO)")
    print(f"Val: {len(val)} (hardhat only, clean)")
    
    return train, val

train_samples, val_samples = combine_datasets(hardhat_samples, coco_samples)

# =============================================================================
# CELL 8: Create YOLO Dataset
# =============================================================================

def write_yolo(train, val):
    print("\n" + "="*50)
    print("📁 WRITING YOLO DATASET")
    print("="*50)
    
    for d in [TRAIN_DIR/"images", TRAIN_DIR/"labels", VAL_DIR/"images", VAL_DIR/"labels"]:
        d.mkdir(parents=True, exist_ok=True)
    
    def write(samples, name, out_dir):
        counts = {0: 0, 1: 0}
        for i, s in enumerate(tqdm(samples, desc=name)):
            img = s["image_path"]
            dst_img = out_dir / "images" / f"{name}_{i:05d}{img.suffix}"
            dst_lbl = out_dir / "labels" / f"{name}_{i:05d}.txt"
            
            try:
                shutil.copy2(img, dst_img)
            except:
                continue
            
            with open(dst_lbl, 'w') as f:
                for o in s["objects"]:
                    b = o["bbox"]
                    f.write(f"{o['class_id']} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                    counts[o["class_id"]] += 1
        return counts
    
    t = write(train, "train", TRAIN_DIR)
    v = write(val, "val", VAL_DIR)
    
    print(f"\n✅ Train: helmet={t[0]}, no_helmet={t[1]}")
    print(f"✅ Val: helmet={v[0]}, no_helmet={v[1]}")

write_yolo(train_samples, val_samples)

# =============================================================================
# CELL 9: Dataset YAML
# =============================================================================

yaml_content = {
    'path': str(FINAL_DATASET),
    'train': 'train/images',
    'val': 'valid/images',
    'names': {0: 'helmet', 1: 'no_helmet'}
}

yaml_path = FINAL_DATASET / "dataset.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f)

print(f"✅ Dataset YAML: {yaml_path}")

# =============================================================================
# CELL 10: Train
# =============================================================================

import os
import sys

# Disable integrations that can cause hangs
os.environ['WANDB_DISABLED'] = 'true'
os.environ['COMET_MODE'] = 'disabled'

# Flush output
sys.stdout.flush()

print("🚀 Starting training...")
sys.stdout.flush()

# Use subprocess to run training (more reliable in notebooks)
import subprocess
cmd = [
    sys.executable, '-m', 'ultralytics.yolo.v8.detect.train',
    '--data', str(yaml_path),
    '--epochs', '40',
    '--patience', '10', 
    '--batch', '16',
    '--imgsz', '640',
    '--device', '0',
    '--workers', '0',
    '--cache', 'True',
    '--project', 'helmet_training',
    '--name', 'yolov8s_run',
    '--exist_ok', 'True',
]

# Alternative: Direct Python API with all fixes
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data=str(yaml_path),
    epochs=40,
    patience=10,
    batch=16,
    imgsz=640,
    device=0,
    workers=0,
    cache=False,  # Disable cache to avoid memory issues
    amp=False,    # Disable AMP - can cause hangs
    optimizer='AdamW',
    lr0=0.01,
    warmup_epochs=3,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
    degrees=15, translate=0.2, scale=0.5,
    shear=10, perspective=0.002,
    mosaic=1.0, mixup=0.2, copy_paste=0.15,
    close_mosaic=10,
    project='helmet_training',
    name='yolov8s_run',
    exist_ok=True,
    seed=42,
)
print("✅ Training complete!")

# =============================================================================
# CELL 11: Evaluate & Export
# =============================================================================

best_path = Path('helmet_training') / 'yolov8s_run' / 'weights' / 'best.pt'
best_model = YOLO(str(best_path))

# Validate
val_results = best_model.val(data=str(yaml_path))

metrics = {
    'mAP50': float(val_results.box.map50),
    'mAP50-95': float(val_results.box.map),
    'precision': float(val_results.box.mp),
    'recall': float(val_results.box.mr),
}

if len(val_results.box.r) > 1:
    metrics['no_helmet_recall'] = float(val_results.box.r[1])

print(f"\n📊 mAP@50: {metrics['mAP50']:.4f}")
print(f"📊 no_helmet recall: {metrics.get('no_helmet_recall', 0):.4f}")

# Export
export_dir = WORK_DIR / 'exports'
export_dir.mkdir(exist_ok=True)

shutil.copy2(best_path, export_dir / 'helmet_yolov8s_best.pt')
best_model.export(format='onnx', imgsz=640, simplify=True)

# Baseline stats for drift detection
def baseline_stats(img_dir, n=500):
    b, c = [], []
    for p in list(img_dir.glob("*"))[:n]:
        img = cv2.imread(str(p))
        if img is None: continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b.append(float(np.mean(g)))
        c.append(float(np.std(g)))
    return {"brightness_mean": np.mean(b), "brightness_std": np.std(b),
            "contrast_mean": np.mean(c), "contrast_std": np.std(c)}

with open(export_dir / 'baseline_stats.json', 'w') as f:
    json.dump(baseline_stats(TRAIN_DIR / "images"), f, indent=2)

with open(export_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

shutil.make_archive(str(WORK_DIR / "helmet_exports"), 'zip', export_dir)

print(f"\n🎉 DONE!")
print(f"📥 Download: /kaggle/working/helmet_exports.zip")
