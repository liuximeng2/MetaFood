import os
import cv2
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np

# === Configuration ===
model_type = "vit_h"
checkpoint_path = "./SAM_ckpt/sam_vit_h_4b8939.pth"
input_dir = "./food_combos"
output_dir = "./output_segments"
MAX_IMAGE_SIZE = 1024  # Maximum size for resizing images
MIN_MASK_AREA = 1000  # Minimum area for masks
TOP_N = 5  # Number of top masks to keep
CENTRE_BIAS_WEIGHT = 0.6 # Weight for center bias in scoring

os.makedirs(output_dir, exist_ok=True)

# === Load SAM Model ===
device = "cuda:2" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # smaller = fewer masks, faster
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
)

# === Process Subfolders ===
for subfolder in tqdm(os.listdir(input_dir)):
    subfolder_path = os.path.join(input_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        continue

    image_file = image_files[0]
    image_path = os.path.join(subfolder_path, image_file)

    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    cx, cy = w / 2, h / 2  # image center

    scored_masks = []
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8)
        area = np.sum(seg)
        if area < MIN_MASK_AREA:
            continue

        # Centroid of mask
        ys, xs = np.where(seg > 0)
        if len(xs) == 0: continue
        mx, my = np.mean(xs), np.mean(ys)
        dist = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
        norm_dist = dist / np.sqrt(cx ** 2 + cy ** 2)  # normalize to [0,1]

        # Scoring: area and center closeness
        score = (1 - CENTRE_BIAS_WEIGHT) * area - CENTRE_BIAS_WEIGHT * norm_dist * area
        scored_masks.append((score, mask))

    # Sort and select top N
    top_masks = sorted(scored_masks, key=lambda x: x[0], reverse=True)[:TOP_N]

    scene_outdir = os.path.join(output_dir, subfolder)
    os.makedirs(scene_outdir, exist_ok=True)

    for i, (score, mask) in enumerate(top_masks):
        seg = mask["segmentation"].astype(np.uint8) * 255
        x, y, w_box, h_box = cv2.boundingRect(seg)

        masked_img = cv2.bitwise_and(image, image, mask=seg)
        cropped = masked_img[y:y+h_box, x:x+w_box]

        out_path = os.path.join(scene_outdir, f"{subfolder}_object_{i}.jpg")
        cv2.imwrite(out_path, cropped)

print("✅ Done! Center-weighted top segments saved in:", output_dir)