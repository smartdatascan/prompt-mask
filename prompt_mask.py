import os
import fire
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from typing import Optional
from transformers import (
    pipeline,
    SamModel,
    SamProcessor
)


def apply_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a red overlay to the masked regions of an image.

    Args:
        image (np.ndarray): Input BGR image (H, W, 3)
        mask (np.ndarray): Binary mask (H, W) with 255 for object, 0 for background

    Returns:
        np.ndarray: Overlayed image (H, W, 3)
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}")
    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]  # Red overlay (BGR)
    return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)


def generate_masks(
    images_dir: str,
    output_mask_dir: str,
    prompt: str,
    output_visualization_dir: Optional[str] = None,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Generate binary masks from text prompt using Grounding DINO + SAM.

    Args:
        images_dir (str): Folder with input RGB images
        output_mask_dir (str): Where to save binary masks (.png)
        prompt (str): Text prompt describing the object (e.g. "plush dog . toy dog .")
        output_visualization_dir (str, optional): Folder to save visualizations (overlay images)
        threshold (float): Detection threshold for Grounding DINO
        device (str): "cuda" or "cpu" — auto-selected by default
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    if output_visualization_dir:
        os.makedirs(output_visualization_dir, exist_ok=True)

    prompt_string = prompt.strip()
    if not prompt_string.endswith("."):
        prompt_string += " ."

    # Load models
    grounding_dino = pipeline(
        model="IDEA-Research/grounding-dino-tiny",
        task="zero-shot-object-detection",
        device=0 if device == "cuda" else -1
    )
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    for fname in tqdm(os.listdir(images_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(images_dir, fname)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Detect objects
        detections = grounding_dino(
            image,
            candidate_labels=[prompt_string],
            threshold=threshold
        )

        if not detections:
            print(f"[!] No detections for prompt in {fname}")
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        else:
            boxes = [
                [det["box"]["xmin"], det["box"]["ymin"], det["box"]["xmax"], det["box"]["ymax"]]
                for det in detections
            ]

            sam_inputs = sam_processor(
                images=image,
                input_boxes=[boxes],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)

            masks_list = sam_processor.post_process_masks(
                masks=sam_outputs.pred_masks,
                original_sizes=sam_inputs.original_sizes,
                reshaped_input_sizes=sam_inputs.reshaped_input_sizes
            )

            masks = masks_list[0]  # shape: (1, N, H, W)

            if masks.ndim == 4 and masks.shape[0] == 1:
                masks = masks.squeeze(0)

            if masks.numel() == 0:
                print(f"[!] Empty masks for {fname}")
                mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            else:
                mask = torch.any(masks > 0, dim=0).to(torch.uint8) * 255
                mask = mask.cpu().numpy()

        if mask.shape != image_np.shape[:2]:
            print(f"[!] Skipping {fname}: mask shape {mask.shape} != image shape {image_np.shape[:2]}")
            continue

        mask_path = os.path.join(output_mask_dir, os.path.splitext(fname)[0] + "_mask.png")
        cv2.imwrite(mask_path, mask)

        if output_visualization_dir:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            overlay_img = apply_overlay(image_bgr, mask)
            overlay_path = os.path.join(output_visualization_dir, fname)
            cv2.imwrite(overlay_path, overlay_img)

    print("✅ All masks saved to:", output_mask_dir)
    if output_visualization_dir:
        print("🖼️ Visualizations saved to:", output_visualization_dir)


if __name__ == "__main__":
    fire.Fire(generate_masks)
