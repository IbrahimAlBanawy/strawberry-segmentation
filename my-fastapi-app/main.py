from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
from pathlib import Path
import zipfile


# Run this code
# Then open the terminal and write: uvicorn test:app --reload



app = FastAPI()

# Load model once on startup
model = YOLO(r"C:\Users\Ibrahim_Hegazi\Desktop\Graduation_Project\ModelOutput After Training on Kaggle Notebooks\kaggle\working\strawberry_seg\weights\best.pt")


# Temp output directory
BASE_OUTPUT_DIR = "outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    # Save uploaded file
    input_bytes = await file.read()
    nparr = np.frombuffer(input_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Create unique output directory
    request_id = str(uuid.uuid4())
    output_dir = os.path.join(BASE_OUTPUT_DIR, request_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save input image temporarily
    input_path = os.path.join(output_dir, "input.jpg")
    cv2.imwrite(input_path, img)

    # Run inference
    results = model.predict(source=input_path, save=False)

    # Process and save outputs
    for result in results:
        annotated_img = result.plot()
        annotated_path = os.path.join(output_dir, "annotated.jpg")
        cv2.imwrite(annotated_path, annotated_img[:, :, ::-1])

        # Save masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                mask_path = os.path.join(output_dir, f"mask_{i}.png")
                cv2.imwrite(mask_path, (mask * 255).astype('uint8'))

        # Save labels
        if result.boxes is not None:
            labels = result.boxes.data.cpu().numpy()
            with open(os.path.join(output_dir, "labels.txt"), 'w') as f:
                for *xyxy, conf, cls in labels:
                    f.write(f"{int(cls)} {conf:.2f} {' '.join(map(str, xyxy))}\n")

    # Zip results
    zip_path = os.path.join(BASE_OUTPUT_DIR, f"{request_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in Path(output_dir).glob("*"):
            zipf.write(file, arcname=file.name)



    return FileResponse(zip_path, media_type='application/zip', filename="segmentation_results.zip")
