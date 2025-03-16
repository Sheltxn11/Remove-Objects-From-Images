import sys
import os
import base64
import io

sys.path.insert(0, os.path.abspath('./detectron2'))

import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import StableDiffusionInpaintPipeline
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog


#Initialise Variables for later use
predictor = None
cfg = None
pipe = None

def initialize_models():

    global predictor, cfg, pipe
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

def get_plant_mask(image):
    global predictor, cfg
    
    if predictor is None or cfg is None:
        initialize_models()

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
            plant_mask = cv2.bitwise_or(plant_mask, (pred_masks[i] * 255).astype(np.uint8))
        
        kernel_dilate = np.ones((7, 7), np.uint8)
        plant_mask = cv2.dilate(plant_mask, kernel_dilate, iterations=2)
        kernel_close = np.ones((11, 11), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_close)
        return plant_mask
    return None

def decode_image_data(image_data: str) -> np.ndarray:
    """
    Decode base64 image data into a numpy array.
    
    Args:
        image_data (str): Base64 encoded image data, with or without data URI prefix
        
    Returns:
        np.ndarray: Decoded image as a numpy array in BGR format
        
    Raises:
        ValueError: If image data cannot be decoded
    """
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Error decoding image data")
    
    return image

def prepare_images_for_inpainting(image: np.ndarray, mask: np.ndarray) -> tuple[Image.Image, Image.Image, int, int]:
    """
    Prepare images for the inpainting model by converting formats and ensuring correct dimensions.
    
    Args:
        image (np.ndarray): Input image in BGR format
        mask (np.ndarray): Binary mask image
        
    Returns:
        tuple: (PIL image, PIL mask, target width, target height)
    """
    original_height, original_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask)
    
    # Done this to keep the original resolution
    target_width = (original_width // 8) * 8
    target_height = (original_height // 8) * 8
    
    image_pil = image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    mask_pil = mask_pil.resize((target_width, target_height), Image.Resampling.NEAREST)
    
    return image_pil, mask_pil, target_width, target_height

def encode_result_image(result_image: Image.Image, original_size: tuple[int, int]) -> str:
    """
    Encode the result image as a base64 string.
    
    Args:
        result_image (Image.Image): The processed image
        original_size (tuple): Original image dimensions (width, height)
        
    Returns:
        str: Base64 encoded image with data URI prefix
    """
    if result_image.size != original_size:
        result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG", quality=100)
    result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{result_base64}"

def run_inpainting(image_pil: Image.Image, mask_pil: Image.Image, 
                  target_height: int, target_width: int) -> Image.Image:
    """
    Run the stable diffusion inpainting model.
    
    Args:
        image_pil (Image.Image): Input image
        mask_pil (Image.Image): Mask image
        target_height (int): Target height for processing
        target_width (int): Target width for processing
        
    Returns:
        Image.Image: The inpainted result image
    """
    prompt = "clean interior"
    negative_prompt = "plants, trees, flowers, poor quality, blurry, plantlings, no extra designs"
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=20,
        guidance_scale=7.5,
        height=target_height,
        width=target_width
    ).images[0]

def gan_inpaint(image_data: str) -> dict:
    """
    Remove plants from an image using GAN-based inpainting.
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        dict: Contains either:
            - {'success': True, 'processed_image': base64_image_data}
            - {'error': error_message}
    """
    global pipe
    
    try:
        if pipe is None:
            initialize_models()
        
        image = decode_image_data(image_data)
        original_size = (image.shape[1], image.shape[0])
        
        mask = get_plant_mask(image)
        if mask is None:
            return {"error": "No plants detected in the image."}
        
        image_pil, mask_pil, target_width, target_height = prepare_images_for_inpainting(image, mask)
        
        result = run_inpainting(image_pil, mask_pil, target_height, target_width)
        
        result_base64 = encode_result_image(result, original_size)
        
        return {
            "success": True,
            "processed_image": result_base64
        }
        
    except Exception as e:
        return {"error": str(e)}

initialize_models()

if __name__ == "__main__":
    with open("test.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        result = gan_inpaint(image_data) 