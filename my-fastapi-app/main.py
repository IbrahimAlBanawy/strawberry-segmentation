# main.py

from fastapi import FastAPI, UploadFile, File, Response, Depends, HTTPException, Security # Added Depends, HTTPException, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader # Added APIKeyHeader
from ultralytics import YOLO
import cv2
import numpy as np
import io
import os
import uuid
from pathlib import Path
import zipfile
import shutil # Added for cleanup in shutdown event

# --- 1. Initialize the FastAPI instance ---
app = FastAPI(
    title="YOLOv11Seg Inference API",
    description="API for performing segmentation with YOLOv11Seg model.",
    version="1.0.0"
)

# --- API Key Configuration ---
# WARNING: Hardcoding API keys is NOT recommended for production environments.
# This is for demonstration purposes only. For production, use environment variables.
API_KEY = "YOLO_SEG_API_KEY_!@#$abcDEF1234567890" # Replace with your desired public key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency function to validate the API key.
    Checks if the provided API key matches the hardcoded one.
    """
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials: Invalid API Key")


# --- 2. Model Loading and Temporary Directory Setup ---
# Global variable to hold the model instance
model: YOLO = None # Type hint for clarity
BASE_OUTPUT_DIR = "outputs" # Directory to store temporary output files

@app.on_event("startup")
async def load_model_and_create_dirs():
    """
    Loads the YOLOv11Seg model and creates the temporary output directory
    when the FastAPI application starts.
    """
    global model
    try:
        # Load the YOLOv11Seg model from the specified path
        # Ensure 'models/best.pt' is accessible in your deployment environment
        model = YOLO("models/best.pt")
        print("YOLOv11Seg model loaded successfully from models/best.pt.")
    except Exception as e:
        print(f"Error loading YOLOv11Seg model: {e}")
        # Depending on criticality, you might want to raise an exception
        # or set a flag to indicate model loading failure.

    # Create the base output directory if it doesn't exist
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Temporary output directory '{BASE_OUTPUT_DIR}' ensured.")

@app.on_event("shutdown")
async def cleanup_output_dir():
    """
    Cleans up the temporary output directory when the FastAPI application shuts down.
    This helps prevent accumulation of temporary files.
    """
    if os.path.exists(BASE_OUTPUT_DIR):
        try:
            shutil.rmtree(BASE_OUTPUT_DIR)
            print(f"Temporary output directory '{BASE_OUTPUT_DIR}' cleaned up.")
        except Exception as e:
            print(f"Error cleaning up output directory '{BASE_OUTPUT_DIR}': {e}")


# --- 3. Define the Prediction Endpoint ---
@app.post("/segment/") # Changed endpoint path
async def segment_image(file: UploadFile = File(...), api_key: str = Depends(get_api_key)): # Added API key dependency
    """
    Receives an image, performs segmentation using the YOLOv11Seg model,
    and returns a ZIP file containing the annotated image, masks, and labels.
    Requires an API key in the 'X-API-Key' header.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type. Please upload an image."}, status_code=400)

    # Ensure the model is loaded before proceeding
    if model is None:
        return JSONResponse(content={"error": "Model not loaded. Server is not ready."}, status_code=503)

    try:
        # --- 4. Take in the input & Preprocess the image ---
        # Read the image bytes from the uploaded file
        input_bytes = await file.read()

        # Convert bytes to a NumPy array and then decode using OpenCV
        nparr = np.frombuffer(input_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Reads image as BGR

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode image. Invalid image format or corrupted data."})

        # Generate a unique ID for this request to manage temporary files
        request_id = str(uuid.uuid4())
        output_dir = os.path.join(BASE_OUTPUT_DIR, request_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save the input image temporarily (optional, but useful for debugging/logging)
        input_path = os.path.join(output_dir, "input.jpg")
        cv2.imwrite(input_path, img) # Saves the input image

        # --- 5. Send it to the model (YOLOv11Seg Inference) ---
        # Perform inference using the loaded YOLO model
        # 'source' can be a file path, so we use the temporarily saved input_path
        results = model.predict(source=input_path, save=False) # save=False to handle saving manually

        # --- 6. Process and Format the output to be sent back to the backend ---
        # Iterate through results (there might be multiple if batch processing)
        for result in results:
            # Get the annotated image (with bounding boxes and masks drawn)
            # result.plot() returns a NumPy array in BGR format (OpenCV default)
            annotated_img = result.plot()
            annotated_path = os.path.join(output_dir, "annotated.jpg")
            # Save the annotated image. OpenCV's imwrite expects BGR.
            cv2.imwrite(annotated_path, annotated_img)

            # Extract and save masks if available
            if result.masks is not None:
                # masks.data is a tensor, move to CPU and convert to NumPy
                masks = result.masks.data.cpu().numpy()
                for i, mask in enumerate(masks):
                    mask_path = os.path.join(output_dir, f"mask_{i}.png")
                    # Convert mask (0-1 float) to 0-255 uint8 for saving as image
                    cv2.imwrite(mask_path, (mask * 255).astype('uint8'))

            # Extract and save detection labels if available
            if result.boxes is not None:
                # boxes.data is a tensor, move to CPU and convert to NumPy
                labels = result.boxes.data.cpu().numpy()
                labels_path = os.path.join(output_dir, "labels.txt")
                with open(labels_path, 'w') as f:
                    for *xyxy, conf, cls in labels:
                        # Format: class_id confidence x_min y_min x_max y_max
                        f.write(f"{int(cls)} {conf:.2f} {' '.join(map(str, xyxy))}\n")

        # Create a ZIP file containing all generated outputs for this request
        zip_path = os.path.join(BASE_OUTPUT_DIR, f"{request_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_in_dir in Path(output_dir).glob("*"):
                # Add each file to the zip archive, keeping its original name
                zipf.write(file_in_dir, arcname=file_in_dir.name)

        # Return the ZIP file as a FileResponse
        return FileResponse(zip_path, media_type='application/zip', filename="segmentation_results.zip")

    except Exception as e:
        print(f"Error during segmentation: {e}")
        # Return a JSON error response for internal server errors
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)

# --- How to run your FastAPI app locally (for testing): ---
# Save this code as `main.py`.
# Install necessary libraries:
# pip install fastapi uvicorn pillow opencv-python numpy ultralytics
#
# Note: For 'ultralytics', you might also need a PyTorch installation.
# Refer to Ultralytics documentation for exact installation steps:
# https://docs.ultralytics.com/
#
# Run from your terminal:
# uvicorn main:app --host 0.0.0.0 --port 8000
#
# You can then access the API documentation at http://0.0.0.0:8000/docs
# and test the endpoint. You will need to provide the 'X-API-Key' header
# with the value "YOUR_HARDCODED_PUBLIC_API_KEY_HERE" (or whatever you set it to).

# --- Railway Deployment Considerations ---
# For Railway, you'll need a `requirements.txt` file:
# fastapi
# uvicorn
# pillow # Pillow is still useful for general image handling, though not directly used in the main flow here
# opencv-python
# numpy
# ultralytics
# # Add your PyTorch dependency if not included by ultralytics automatically, e.g.:
# # torch
#
# Your Railway project would typically be configured to run `uvicorn main:app --host 0.0.0.0 --port $PORT`
# where `$PORT` is an environment variable provided by Railway.
#
# Ensure your 'models/best.pt' file is present in the 'models' directory at the root
# of your deployed application on Railway. You might need to configure your Railway
# deployment to include this directory and its contents.
