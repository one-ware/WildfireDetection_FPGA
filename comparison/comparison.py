import os
import argparse
import sys

# ==========================================
# ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()

parser.add_argument("--threads", type=int, default=1,
                    help="Number of threads for inference")

parser.add_argument("--model", type=str, default="all",
                    choices=["oneai", "unet", "deeplab", "all"],
                    help="Model to run")

parser.add_argument("--ops", action="store_true",
                    help="Only calculate theoretical operations and exit")

# NEW: confidence thresholds per model
parser.add_argument("--thr_oneai", type=float, default=0.5,
                    help="Confidence threshold for CustomCNN")

parser.add_argument("--thr_unet", type=float, default=0.5,
                    help="Confidence threshold for UNet")

parser.add_argument("--thr_deeplab", type=float, default=0.5,
                    help="Confidence threshold for DeepLabV3+")

args = parser.parse_args()

NUM_THREADS = args.threads
MODEL_TYPE = args.model

THRESHOLDS = {
    "CustomCNN": args.thr_oneai,
    "UNet": args.thr_unet,
    "DeepLabV3+": args.thr_deeplab
}

# ==========================================
# ENV THREAD CONTROL
# ==========================================
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ONEAI_MODEL_PATH = os.path.join(BASE_DIR, "../models/one_ai_seg.tflite")
UNET_MODEL_PATH = os.path.join(BASE_DIR, "../models/unet_seg.onnx")
DEEPLAB_MODEL_PATH = os.path.join(BASE_DIR, "../models/deeplab_binary.onnx")
INPUT_VIDEO = os.path.join(BASE_DIR, "../video/demo_video.mp4")

# ==========================================
# FLOPS CALCULATION (ONNX)
# ==========================================
def calculate_onnx_flops(model_path):
    import onnx
    from onnx import shape_inference

    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    total_flops = 0
    total_params = 0
    initializer_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "Conv":
            continue

        weight_name = node.input[1]
        if weight_name not in initializer_map:
            continue

        weight = initializer_map[weight_name]
        Cout, Cin, kH, kW = weight.dims
        total_params += Cout * Cin * kH * kW

        for value_info in graph.value_info:
            if value_info.name == node.output[0]:
                shape = value_info.type.tensor_type.shape
                dims = [d.dim_value for d in shape.dim]
                if len(dims) == 4:
                    _, _, H, W = dims
                    flops = 2 * H * W * Cin * Cout * kH * kW
                    total_flops += flops
                break

    print(f"Parameters: {total_params/1e6:.2f} M")
    print(f"FLOPs: {total_flops/1e9:.2f} GFLOPs")


def calculate_tflite_stats(model_path):
    """Calculate FLOPs and parameters for TFLite model"""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available, showing file size only")
        file_size = os.path.getsize(model_path)
        print(f"Model size: {file_size/1e6:.2f} MB")
        return
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    total_flops = 0
    total_params = 0
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    
    # Create a map of tensor indices to their shapes
    tensor_map = {t['index']: t for t in tensor_details}
    
    # Iterate through all operations in the model
    for op_index in range(len(interpreter._get_ops_details())):
        op_detail = interpreter._get_ops_details()[op_index]
        op_name = op_detail['op_name']
        
        inputs = op_detail['inputs']
        outputs = op_detail['outputs']
        
        if len(inputs) == 0 or len(outputs) == 0:
            continue
            
        # Get output shape
        if outputs[0] in tensor_map:
            output_shape = tensor_map[outputs[0]]['shape']
        else:
            continue
        
        # Calculate FLOPs based on operation type
        if op_name == 'CONV_2D':
            # Conv2D: output_h * output_w * kernel_h * kernel_w * in_channels * out_channels * 2
            if len(inputs) >= 2 and inputs[1] in tensor_map:
                weight_shape = tensor_map[inputs[1]]['shape']
                # TFLite Conv2D weights: [out_channels, kernel_h, kernel_w, in_channels]
                if len(weight_shape) == 4:
                    Cout, kH, kW, Cin = weight_shape
                    total_params += Cout * kH * kW * Cin
                    
                    if len(output_shape) == 4:
                        _, H, W, _ = output_shape
                        flops = 2 * H * W * Cin * Cout * kH * kW
                        total_flops += flops
        
        elif op_name == 'DEPTHWISE_CONV_2D':
            # DepthwiseConv2D: output_h * output_w * kernel_h * kernel_w * channels * 2
            if len(inputs) >= 2 and inputs[1] in tensor_map:
                weight_shape = tensor_map[inputs[1]]['shape']
                # TFLite DepthwiseConv2D weights: [1, kernel_h, kernel_w, channels]
                if len(weight_shape) == 4:
                    _, kH, kW, C = weight_shape
                    total_params += kH * kW * C
                    
                    if len(output_shape) == 4:
                        _, H, W, _ = output_shape
                        flops = 2 * H * W * C * kH * kW
                        total_flops += flops
        
        elif op_name in ['FULLY_CONNECTED', 'DENSE']:
            # Fully connected: input_size * output_size * 2
            if len(inputs) >= 2 and inputs[1] in tensor_map:
                weight_shape = tensor_map[inputs[1]]['shape']
                if len(weight_shape) == 2:
                    input_size, output_size = weight_shape
                    total_params += input_size * output_size
                    flops = 2 * input_size * output_size
                    total_flops += flops
    
    print(f"Parameters: {total_params/1e6:.2f} M")
    print(f"FLOPs: {total_flops/1e9:.2f} GFLOPs")


# ==========================================
# OPS MODE
# ==========================================
if args.ops:
    print("\n===== Theoretical Operations =====")
    #ONEAI_MODEL_PATH = os.path.join(BASE_DIR, "../models/one_ai.onnx")

    if MODEL_TYPE in ["oneai", "all"]:
        print("\nCustomCNN:")
        if ONEAI_MODEL_PATH.endswith('.onnx'):
            calculate_onnx_flops(ONEAI_MODEL_PATH)
        else:
            calculate_tflite_stats(ONEAI_MODEL_PATH)

    if MODEL_TYPE in ["unet", "all"]:
        print("\nUNet:")
        if UNET_MODEL_PATH.endswith('.onnx'):
            calculate_onnx_flops(UNET_MODEL_PATH)
        else:
            calculate_tflite_stats(UNET_MODEL_PATH)

    if MODEL_TYPE in ["deeplab", "all"]:
        print("\nDeepLabV3+ MobileNet:")
        if DEEPLAB_MODEL_PATH.endswith('.onnx'):
            calculate_onnx_flops(DEEPLAB_MODEL_PATH)
        else:
            calculate_tflite_stats(DEEPLAB_MODEL_PATH)

    sys.exit(0)

# ==========================================
# IMPORT RUNTIME LIBS
# ==========================================
import cv2
import time
import numpy as np
import threading
import queue
import onnxruntime as ort

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("Warning: TensorFlow not available. TFLite models will not work.")

# ==========================================
# PIPELINE
# ==========================================
def run_pipeline(model_path, label, threshold):

    print(f"\n===== Running {label} with {NUM_THREADS} threads =====")
    print(f"Confidence threshold: {threshold}")

    # Detect model format
    is_tflite = model_path.endswith('.tflite')
    is_onnx = model_path.endswith('.onnx')

    if is_tflite:
        if not TFLITE_AVAILABLE:
            print(f"Error: TensorFlow not available for TFLite model: {model_path}")
            return
        
        print(f"Using TFLite interpreter")
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=NUM_THREADS
        )
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        input_shape = input_details['shape']
        _, H, W, _ = input_shape  # TFLite typically uses NHWC format
        input_index = input_details['index']
        output_index = output_details['index']
        
        session = None
        input_name = None
        
    elif is_onnx:
        print(f"Using ONNX Runtime")
        so = ort.SessionOptions()
        so.intra_op_num_threads = NUM_THREADS
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        input_meta = session.get_inputs()[0]
        _, _, H, W = input_meta.shape
        input_name = input_meta.name
        
        interpreter = None
        input_index = None
        output_index = None
    else:
        print(f"Error: Unknown model format for {model_path}. Expected .onnx or .tflite")
        return

    frame_queue = queue.Queue(maxsize=3)
    result_queue = queue.Queue(maxsize=3)

    def reader():
        cap = cv2.VideoCapture(INPUT_VIDEO)
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            frame_queue.put(frame)
        cap.release()

    def inference_worker():
        while True:
            frame = frame_queue.get()
            if frame is None:
                result_queue.put(None)
                break

            img = cv2.resize(frame, (W, H))
            img = img.astype(np.float32) / 255.0
            
            start = time.time()
            
            if is_tflite:
                # TFLite expects NHWC format (batch, height, width, channels)
                img = np.expand_dims(img, axis=0)
                interpreter.set_tensor(input_index, img)
                interpreter.invoke()
                output = interpreter.get_tensor(output_index)
            else:
                # ONNX expects NCHW format (batch, channels, height, width)
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                output = session.run(None, {input_name: img})
                output = output[0]
            
            inf_time = (time.time() - start) * 1000

            result_queue.put((frame, output, inf_time))

    def writer():
        cap = cv2.VideoCapture(INPUT_VIDEO)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out = cv2.VideoWriter(
            os.path.join(BASE_DIR, f"output_{label}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        total_time = 0
        frame_count = 0
        # 10-second reporting
        last_report_time = time.time()
        interval_inf_time = 0.0
        interval_inf_count = 0

        print(f"[Writer-{label}] started, reporting avg inference every 10s")

        while True:
            item = result_queue.get()
            if item is None:
                break

            frame, mask, inf_time = item

            total_time += inf_time
            frame_count += 1
            interval_inf_time += inf_time
            interval_inf_count += 1

            mask = np.squeeze(mask).astype(np.float32)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            colored = np.zeros_like(frame)
            colored[:, :, 2] = (mask > threshold) * 255

            overlay = cv2.addWeighted(frame, 0.7, colored, 0.3, 0)

            fps_inst = 1000.0 / inf_time if inf_time > 0 else 0

            cv2.putText(
                overlay,
                f"{label}: {fps_inst:.2f} FPS",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (209, 255, 0),
                2
            )

            out.write(overlay)

            # report every 10 seconds
            now_report = time.time()
            if now_report - last_report_time >= 10.0:
                if interval_inf_count > 0:
                    avg_inf_ms = interval_inf_time / interval_inf_count
                    fps_10s = 1000.0 / avg_inf_ms if avg_inf_ms > 0 else 0
                    print(f"[{label}] Avg inference last 10s: {avg_inf_ms:.1f} ms/frame ({fps_10s:.2f} FPS)")
                else:
                    print(f"[{label}] No inference frames in last 10s")
                interval_inf_time = 0.0
                interval_inf_count = 0
                last_report_time = now_report

        out.release()

        avg = total_time / frame_count
        print(f"\n[{label}] Final Avg: {avg:.2f} ms")
        print(f"[{label}] Final Model FPS: {1000/avg:.2f}")

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=inference_worker)
    t3 = threading.Thread(target=writer)

    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()


# ==========================================
# MAIN
# ==========================================
if MODEL_TYPE == "all":
    run_pipeline(ONEAI_MODEL_PATH, "CustomCNN", THRESHOLDS["CustomCNN"])
    run_pipeline(DEEPLAB_MODEL_PATH, "DeepLabV3+", THRESHOLDS["DeepLabV3+"])
    run_pipeline(UNET_MODEL_PATH, "UNet", THRESHOLDS["UNet"])

elif MODEL_TYPE == "oneai":
    run_pipeline(ONEAI_MODEL_PATH, "CustomCNN", THRESHOLDS["CustomCNN"])

elif MODEL_TYPE == "unet":
    run_pipeline(UNET_MODEL_PATH, "UNet", THRESHOLDS["UNet"])

elif MODEL_TYPE == "deeplab":
    run_pipeline(DEEPLAB_MODEL_PATH, "DeepLabV3+", THRESHOLDS["DeepLabV3+"])