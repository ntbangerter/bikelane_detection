from transformers import ViTImageProcessor, ViTForImageClassification

def run_image_classification():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', device_map='cuda')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map='cuda')

    def image_classification(img):
        inputs = processor(images=img, return_tensors='pt').to('cuda')
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])

    mjpeg_stream_watcher(
        fn=image_classification,
        imshow=True,
    )
