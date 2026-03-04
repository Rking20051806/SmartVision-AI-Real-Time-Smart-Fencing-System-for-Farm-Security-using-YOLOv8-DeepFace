# SmartVision Multi-Model Detection API (Vercel-ready)

## Features
- `/detect/human`: Detects humans, classifies as child/woman/man, estimates age
- `/detect/animal`: Detects farm animals
- `/detect/insect`: Detects insects/wild animals
- Uses YOLOv8 models (trained on Kaggle datasets)

## How to Use

### 1. Download/Train Models from Kaggle
- Place your `kaggle.json` in the project root or `~/.kaggle/`
- Download datasets or trained models:
  ```
  kaggle datasets download -d <your-dataset-id>
  unzip <downloaded-file.zip>
  # Place yolov8_human.pt, yolov8_animal.pt, yolov8_insect.pt in project root
  ```
- Or train your own YOLOv8 models using Ultralytics:
  ```
  pip install ultralytics
  yolo detect train data=your_data.yaml model=yolov8n.pt imgsz=640 epochs=50
  # Save as yolov8_human.pt, yolov8_animal.pt, yolov8_insect.pt
  ```

### 2. Install Requirements
```
pip install -r requirements.txt
```

### 3. Run Locally
```
uvicorn main_api:app --reload
```

### 4. Deploy to Vercel
- Use Vercel Python serverless function setup (see Vercel docs)
- Make sure model files are included in deployment (or download at startup)

## Example API Usage
- POST image to `/detect/human`, `/detect/animal`, or `/detect/insect`
- Response: JSON with detections (bounding boxes, class, confidence, etc.)

## Example Kaggle Datasets
- Human detection/classification: [CrowdHuman](https://www.kaggle.com/datasets/andrewmvd/crowdhuman)
- Animal detection: [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- Insect detection: [Insect Species Detection](https://www.kaggle.com/datasets/andrewmvd/insect-detection)
- Wild animal detection: [iWildCam](https://www.kaggle.com/c/iwildcam-2020-fgvc7/data)

## Notes
- You must provide your own trained YOLOv8 models for best results.
- For Vercel, keep model files small or download at runtime.
- For custom actions (shock, alert, etc.), add logic in the API response.
