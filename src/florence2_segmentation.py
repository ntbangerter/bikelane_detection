from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image, ImageDraw
import requests
import copy
import torch
import numpy as np

from picamera2_client.client import mjpeg_stream_watcher, fetch_jpeg


model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def run_example(image, task_prompt="<REFERRING_EXPRESSION_SEGMENTATION>", text_input=""):
    prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
        # image_size=image.shape
    )

    return parsed_answer


def classification(image):
    print(run_example(image, "<CAPTION>"))


def mask(image, label="person"):
    # print(image.shape)
    image = Image.fromarray(image)
    response = run_example(image, text_input=label)
    
    polygon = response['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons'][0][0]
    # polygon = np.array(polygon)
    
    # img_mask = Image.new('1', image.shape, 0)
    img_mask = Image.new('1', (image.width, image.height), 0)
    ImageDraw.Draw(img_mask).polygon(polygon, outline=1, fill=1)
    img_mask.save("./test-mask.jpg")
    mask = np.array(img_mask)
    return mask, polygon


def bikelane_detection(label="a car on the street"):
    image = fetch_jpeg(high_res=False)
    image = Image.fromarray(image[..., ::-1])
    image.show()
    response = run_example(image, text_input=label)
    
    polygon = response['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons'][0][0]

    ImageDraw.Draw(image).polygon(polygon, outline="blue", fill="blue")
    image.show()

    
# bikelane_detection()


# mjpeg_stream_watcher(
#     fn=mask,
#     imshow=True,
# )
