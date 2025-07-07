import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import zipfile
from pathlib import Path

# Load model from Hugging Face Hub
model = YOLO("ibrahim/strawberry-segmentation")  # Replace with your model ID


def segment_image(input_img):
    # Create output dir
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save input image
    input_path = os.path.join(output_dir, "input.jpg")
    cv2.imwrite(input_path, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

    # Run inference
    results = model.predict(source=input_path, save=False)

    # Process results
    annotated_img = None
    for result in results:
        annotated_img = result.plot()  # BGR format
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio

    return annotated_img if annotated_img is not None else input_img


# Gradio UI
demo = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Image(label="Segmentation Result"),
    title="Strawberry Segmentation with YOLOv8",
    examples=["strawberry1.jpg", "strawberry2.jpg"],  # Add example images
)

demo.launch()