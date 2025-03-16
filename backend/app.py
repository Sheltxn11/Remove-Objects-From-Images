import sys
import os

sys.path.insert(0, os.path.abspath('./detectron2'))
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def initialize_model():
    """Initialize and return the Detectron2 model configuration and predictor."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg, DefaultPredictor(cfg)

def get_plant_mask(image, predictor, cfg):
    """
    Generate a mask for plants in the image using Detectron2.
    
    Args:
        image (np.ndarray): Input image
        predictor: Detectron2 predictor
        cfg: Model configuration
        
    Returns:
        np.ndarray: Binary mask for plants
    """
    outputs = predictor(image)
    
    coco_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    plant_keywords = ["potted plant", "plant", "tree"]
    plant_class_ids = [i for i, cls in enumerate(coco_classes) 
                      if any(keyword in cls.lower() for keyword in plant_keywords)]
    
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_masks = outputs["instances"].pred_masks.cpu().numpy()
    plant_indices = [i for i, cls_id in enumerate(pred_classes) if cls_id in plant_class_ids]
    
    if len(plant_indices) > 0:
        plant_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in plant_indices:
            mask = (pred_masks[i] * 255).astype(np.uint8)
            kernel = np.ones((15, 15), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)
            plant_mask = cv2.bitwise_or(plant_mask, dilated_mask)
        return plant_mask
    return None

def apply_bilateral_filter(image):
    """Apply bilateral filtering for edge preservation."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def multi_scale_inpaint(image, mask):
    """
    Perform multi-scale inpainting for better results.
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Binary mask
        
    Returns:
        np.ndarray: Inpainted result
    """
    scales = [1.0, 0.8, 0.6, 0.4, 0.2]
    results = []
    weights = [0.4, 0.25, 0.15, 0.12, 0.08]
    
    for scale in scales:
        if scale != 1.0:
            scaled_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            scaled_image = cv2.resize(image, scaled_size, interpolation=cv2.INTER_LANCZOS4)
            scaled_mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image.copy()
            scaled_mask = mask.copy()
        
        inpaint_radius = int(15 / scale) if scale < 1.0 else 15
        result = cv2.inpaint(scaled_image, scaled_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
        result = apply_bilateral_filter(result)
        
        if scale != 1.0:
            result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
        results.append(result)
    
    final_result = np.zeros_like(image, dtype=np.float32)
    for result, weight in zip(results, weights):
        final_result += result.astype(np.float32) * weight
    
    final_result = final_result.astype(np.uint8)
    return apply_bilateral_filter(final_result)

def create_comparison_grid(images, titles, cols=2):
    """Create a grid of images for comparison."""
    rows = (len(images) + cols - 1) // cols
    cell_height = images[0].shape[0]
    cell_width = images[0].shape[1]
    grid = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        i, j = idx // cols, idx % cols
        grid[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = img
        cv2.putText(grid, title, 
                   (j*cell_width + 10, i*cell_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8,
                   (255, 255, 255),
                   2,
                   cv2.LINE_AA)
    
    return grid

def main(image_path):
    """Main function to process the image and generate results."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Initialize model and get mask
    cfg, predictor = initialize_model()
    plant_mask = get_plant_mask(image, predictor, cfg)
    
    if plant_mask is None:
        print("No plants detected in the image.")
        return
    
    # Save mask
    cv2.imwrite("mask.jpg", plant_mask)
    
    # Generate results using different methods
    results = {
        "NS": cv2.inpaint(image, plant_mask, inpaintRadius=3, flags=cv2.INPAINT_NS),
        "TELEA": cv2.inpaint(image, plant_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA),
        "Bilateral": apply_bilateral_filter(cv2.inpaint(image, plant_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)),
        "Multi-scale": multi_scale_inpaint(image, plant_mask)
    }
    
    # Save individual results
    for method, result in results.items():
        cv2.imwrite(f"result_{method.lower()}.jpg", result)
    
    # Create and save comparison grid
    comparison_images = [image] + list(results.values())
    comparison_titles = ["Original"] + list(results.keys())
    comparison_grid = create_comparison_grid(comparison_images, comparison_titles)
    cv2.imwrite("comparison_grid.jpg", comparison_grid)

if __name__ == "__main__":
    main("test.jpg")
