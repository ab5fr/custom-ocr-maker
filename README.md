# Custom OCR Maker

A TensorFlow-based optical character recognition (OCR) pipeline for training and serving custom CAPTCHA text recognition models. The project uses Connectionist Temporal Classification (CTC) loss with a CNN + BiLSTM architecture, powered by the [MLTU](https://github.com/pythonlessons/mltu) library.

## Features

- **End-to-end workflow** — dataset labeling, model training, local evaluation, and REST API inference
- **CTC-based OCR** — handles variable-length text without explicit character segmentation
- **Data augmentation** — brightness, rotation, and erosion/dilation during training
- **ONNX export** — trained models are converted automatically for fast inference
- **Flask API** — serve predictions over HTTP with base64-encoded images

## Project Structure

```
custom-ocr-maker/
├── api.py              # Flask REST API for inference
├── model.py            # CNN + BiLSTM model architecture
├── train.py            # Training script
├── useModel.py         # Local batch evaluation
├── testapi.py          # API client for testing
├── make dataset.py     # Dataset labeling via external OCR service
├── datasets/
│   └── captcha1/       # Labeled training images (filename = label)
├── scraped/            # Raw unlabeled images
└── model/              # Exported model and configs (generated after training)
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- [MLTU](https://pypi.org/project/mltu/) library
- TensorFlow 2.x
- OpenCV, Flask, NumPy, Requests

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ab5fr/custom-ocr-maker.git
   cd custom-ocr-maker
   ```

2. Install dependencies:

   ```bash
   pip install mltu tensorflow opencv-python flask numpy requests onnxruntime
   ```

   > **Note:** MLTU versions can introduce breaking changes. If you encounter compatibility issues, pin a specific version (e.g. `pip install mltu==0.1.4`).

## Usage

### 1. Prepare the Dataset

Place raw CAPTCHA images in the `scraped/` directory. Run the labeling script to send images to an external OCR service and rename them with their predicted labels:

```bash
python "make dataset.py"
```

Labeled images are saved to `datasets/captcha1/` with filenames matching the recognized text (e.g. `Ab3xY9.png`).

> Configure your API key in `make dataset.py` before running.

### 2. Train the Model

```bash
python train.py
```

Training reads images from `datasets/model1/` (update the `captcha_path` in `train.py` to match your dataset directory). The script will:

- Build the vocabulary from dataset labels
- Split data into 90% training / 10% validation
- Train with early stopping, learning rate reduction, and TensorBoard logging
- Export the best model to ONNX format under `models/model1/<timestamp>/`

After training, copy the generated `configs.yaml` and ONNX model into the `model/` directory for inference.

### 3. Evaluate Locally

Run batch inference against a test dataset and report accuracy:

```bash
python useModel.py
```

### 4. Serve via API

Start the Flask inference server:

```bash
python api.py
```

The server listens on `http://0.0.0.0:5555`.

### 5. Test the API

```bash
python testapi.py
```

Place test images in the `testing/` directory before running.

## API Reference

### `POST /predict`

Accepts a JSON body with a base64-encoded image and returns the predicted text.

**Request:**

```json
{
  "image": "<base64-encoded image>"
}
```

**Response:**

```json
{
  "prediction": "Ab3xY9",
  "duration": "12.5ms"
}
```

**Error response:**

```json
{
  "error": "no image"
}
```

## Model Architecture

The recognition model combines:

1. **Convolutional layers** — residual blocks with progressive downsampling to extract visual features
2. **Bidirectional LSTM** — sequence modeling over the feature map
3. **CTC decoder** — maps sequence outputs to the final text prediction

Input images are resized to 150×50 pixels. The vocabulary includes alphanumeric characters (`0-9`, `a-z`, `A-Z`).

## Acknowledgments

Built on top of the [MLTU](https://github.com/pythonlessons/mltu) library by [PyLessons](https://pylessons.com/), which provides training utilities, data pipelines, and ONNX inference helpers for OCR tasks.
