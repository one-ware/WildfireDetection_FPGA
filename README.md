# Wildfire Detection Demo

This repository demonstrates and compares wildfire-segmentation approaches using local ONNX models.

## Structure

- `quartus_project/`: Quartus project folder including vhdl-model.
- `python_uart/`: Python script to run inference on FPGA using PC.
- `model_training/`: Training scripts, ONNX export scripts, and utility scripts.
- `models/`: Exported model files (`.onnx`, `.tflite`).
- `comparison/`: Benchmark and video comparison script, plus generated output videos.
- `video/`: Input video used for inference.
- `dataset/`: Training, validation, and test data.

## Running Model on FPGA

### 1. Programm FPGA

The Quartus project is set for an ACX3000 board with an A3CY100BM16AE7S. Compile quartus project using Quartus 25.3 and programm FPGA.

### 2. Run uart_cnn_visualizer.py

Adapt line 22 of the uart_cnn_visualizer.py script according to the COM port of your board. Then run the script:

```bash
python .\uart_cnn_visualizer.py
```

The script provides further instructions on how to use it. It is possible to either run inference on a video stream (provided by a webcam connected to the PC) or on example images contained in the img/ -folder.


## Running Model Comparison

### 1. Model Comparison and Video Inference

Runs one or multiple ONNX segmentation models on `video/demo_video.mp4`, overlays masks, and writes output videos.

Run all models:

```bash
python comparison/comparison.py --model all --threads 1
```

Run a single model:

```bash
python comparison/comparison.py --model oneai --threads 1
python comparison/comparison.py --model unet --threads 1
python comparison/comparison.py --model deeplab --threads 1
```

Optional thresholds:

```bash
python comparison/comparison.py --model all --thr_oneai 0.5 --thr_unet 0.5 --thr_deeplab 0.5
```

Output:

- `comparison/output_CustomCNN.mp4`
- `comparison/output_UNet.mp4`
- `comparison/output_DeepLabV3+.mp4`

### 2. Theoretical Operations (FLOPs / Parameters)

Calculates theoretical Conv FLOPs and parameter counts from ONNX graph metadata.

```bash
python comparison/comparison.py --ops --model all
```

### 3. Training and ONNX Export

###### UNet training (`wildfire_demo.py`)

```bash
python model_training/wildfire_demo.py --epochs 100 --export_onnx
```

###### DeepLabV3+ MobileNet training (`wildfire_demo_deeplab.py`)

```bash
python model_training/wildfiren_demo_deeplab.py --epochs 300 --export_onnx
```

### 4. Count ONNX Parameters

Counts total parameters from ONNX initializers.

```bash
python model_training/count_onnx_params.py
```

### Requirements

- numpy
- opencv-python
- onnxruntime
- onnxscript
- onnx
- torch
- torchvision
- pillow

Install:

```bash
pip install numpy opencv-python onnxruntime onnx onnxscript torch torchvision pillow
```
