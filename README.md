# Drifters-Anomatrix-AI-2

An AI-powered framework for abnormal event detection and human activity recognition in videos, leveraging deep learning and YOLO models. This project provides tools for training, evaluation, visualization, and API deployment.

## Features

- Abnormal event detection in video frames
- Human activity recognition using YOLOv8
- Visualization tools (heatmaps, confusion matrices, curves)
- API for serving models (`web-test-api`)
- Jupyter notebooks for experimentation and data organization
- Docker support for containerized deployment

## Folder Structure

```
├── app.py, app-L.py, main.py         # Main application scripts
├── Dockerfile                       # Containerization
├── requirements.txt                 # Python dependencies
├── lightning.yaml                   # Lightning configuration
├── abnormal_frames/                 # Abnormal frame detection code
├── models/                          # Trained model weights
├── outputs/, plots/                 # Output images and plots
├── web-test-api/                    # API for model serving
├── notebooks/                       # Jupyter notebooks for research
├── uploaded_videos/, uploads/, videos/ # Video data
├── heatmaps/, logs/, links/         # Supporting data and logs
├── FutureWork/                      # Experimental and future work
```

## Installation

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd Drifters-Anomatrix-AI-2
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Build Docker image:
   ```sh
   docker build -t drifters-anomatrix-ai .
   ```

## Usage

- Run main application:
  ```sh
  python app.py
  ```
- Use the API:
  ```sh
  cd web-test-api
  python app.py
  ```
- Explore notebooks for training, evaluation, and data organization in the `notebooks/` and `FutureWork/` folders.

## Model Information

- YOLOv8 model weights are stored in `models/best_yolov8x.pt`.
- Training and evaluation scripts are provided in notebooks and Python files.

## Visualization

- Output plots (confusion matrix, curves, maps) are in the `plots/` folder.
- Heatmaps and logs are available for deeper analysis.

## API

- The `web-test-api` folder contains a FastAPI app for serving predictions.
- See `web-test-api/requirements.txt` for API dependencies.

## Future Work

- Experimental notebooks and scripts are in the `FutureWork/` folder.
- See `FutureWork/futureWork/` for data download and model improvement ideas.

## Acknowledgements

- YOLOv8 by Ultralytics
- PyTorch Lightning
- Contributors and open-source community
