import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    GroundingDinoProcessor,
    Sam2Processor,
    Sam2Model,
)

from picamera2_client.client import fetch_jpeg


dino_model_id = "IDEA-Research/grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_processor = GroundingDinoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to("cuda")

sam_model_id = "facebook/sam2-hiera-base-plus"
sam_processor = Sam2Processor.from_pretrained(sam_model_id)
sam_model = Sam2Model.from_pretrained(sam_model_id).to("cuda")


CLASS_PROMPT = "car. bicycle. bus. truck. person."
COLOR_MAP = {
    "car": (255, 0, 0), # red
    "bicycle": (0, 255, 0), # green
    "bus": (0, 0, 255), # blue
    "truck": (255, 255, 0), # yellow
    "person": (255, 0, 255), # magenta
}


def apply_mask(image, mask, color):
    color_mask = Image.new("RGBA", image.size, color)
    transparent_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    opaque_mask = Image.composite(color_mask, transparent_mask, mask)
    output = Image.alpha_composite(image, opaque_mask)
    return output


def apply_masks(
    image,
    masks,
    labels,
    alpha=100,
    mask_channel=0,
):
    output = image.copy().convert("RGBA")
    masks = masks.to("cpu")
    for label, instance in zip(labels, masks):
        output = apply_mask(
            output,
            Image.fromarray(np.array(instance[mask_channel])),
            color=COLOR_MAP[label] + (alpha,),
        )

    return output


def run_dino():
    image = fetch_jpeg(high_res=False)
    image = Image.fromarray(image[..., ::-1])

    dino_inputs = dino_processor(
        images=image,
        text=CLASS_PROMPT,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        dino_outputs = dino_model(**dino_inputs)

    results = dino_processor.post_process_grounded_object_detection(
        dino_outputs,
        dino_inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]

    boxes = results["boxes"]
    labels = results["labels"]

    sam_inputs = sam_processor(
        images=[image],
        input_boxes=boxes.unsqueeze(0).to("cpu"),
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"]
    )[0]

    # print(masks.shape)

    masked_image = apply_masks(image, masks, labels)
    return masked_image


def add_timestamp(image, timestamp, position=(10, 10), font_size=30, font_color=(255, 255, 255)):
    # Convert the image to RGBA (if not already)
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    
    # Load a font (change the font path to a valid one on your system)
    font = ImageFont.truetype("arial.ttf", font_size)  # Change to your font path
    draw.text(position, timestamp, font=font, fill=font_color)
    
    return image


def create_video_from_images(images, output_video_path, fps=3):
    # Get dimensions from the first image
    width, height = images[0].size
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'XVID')  # Codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        video_writer.write(image_cv)
    
    # Release the video writer
    video_writer.release()


if __name__ == "__main__":
    images = []
    for i in tqdm(range(100)):
        images.append(run_dino())

    create_video_from_images(
        images,
        "./example_video_3fps.avi",
        fps=3,
    )
    create_video_from_images(
        images,
        "./example_video_2fps.avi",
        fps=2,
    )
    create_video_from_images(
        images,
        "./example_video_1fps.avi",
        fps=1,
    )
