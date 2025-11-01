import numpy as np
import torch
from PIL import Image
import random
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

    apply_masks(image, masks, labels).show()
