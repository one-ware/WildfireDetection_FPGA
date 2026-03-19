#!/usr/bin/env python3
"""
UART CNN Visualizer
Sends 128x128 grayscale images via UART and receives/visualizes CNN results
Supports both classification (digits 0-9) and segmentation
"""

import serial
import serial.tools.list_ports
import cv2
import numpy as np
import threading
import queue
import time
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

# ==================== Configuration ====================
BAUD_RATE = 3000000#3000000
COMPORT = "COM9"
IMAGE_SIZE = 128
CLASS_COUNT = 10
SEGMENTATION_SIZE = (128, 128)
UART_HEADER = bytes([0xFF, 0xAA, 0x55])
RESULT_HEADER = bytes([0xFF, 0xAA, 0x55])  # Same header for received results
QUERY_BYTE = bytes([0x24])
FPS_ESTIMATE = BAUD_RATE / (IMAGE_SIZE * IMAGE_SIZE) / 15

class ResultType(Enum):
    CLASSIFICATION = 1
    SEGMENTATION = 2

@dataclass
class ImagePacket:
    """Container for image to be sent"""
    image: np.ndarray
    timestamp: float

@dataclass
class RawFramePacket:
    """Container for raw received frame data (before processing)"""
    result_type: ResultType
    raw_data: bytearray
    timestamp: float

@dataclass
class ResultPacket:
    """Container for processed CNN results"""
    result_type: ResultType
    data: np.ndarray
    timestamp: float

# ==================== Global State ====================
serial_port: Optional[serial.Serial] = None
send_queue = queue.Queue(maxsize=10)
raw_frame_queue = queue.Queue(maxsize=10)  # Raw bytes from receiver
display_queue = queue.Queue(maxsize=2)
sent_image_queue = queue.Queue(maxsize=2)  # For displaying sent images

webcam_active = False
running = True
running_v = True
first_image = True
#result_mode = ResultType.CLASSIFICATION  # Can be changed to SEGMENTATION
result_mode = ResultType.SEGMENTATION  # Can be changed to CLASSIFICATION

# Threading locks
webcam_lock = threading.Lock()
serial_lock = threading.Lock()

# ==================== UART Functions ====================
def find_uart_port():
    """Automatically find available UART port"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise Exception("No serial ports found!")
    
    # Prefer USB serial ports
    for port in ports:
        if 'USB' in port.description or 'ACM' in port.device or 'USB' in port.device:
            return port.device
    
    # Otherwise return first available
    return ports[0].device

def connect_uart():
    """Establish UART connection"""
    global serial_port
    try:
        port_name = COMPORT#find_uart_port()
        print(f"Connecting to {port_name} at {BAUD_RATE} baud...")
        serial_port = serial.Serial(
            port=port_name,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
        serial_port.set_buffer_size(tx_size = 16000, rx_size = 16000)
        time.sleep(0.5)  # Allow connection to stabilize
        # Flush any existing data
        serial_port.reset_input_buffer()
        serial_port.reset_output_buffer()
        print(f"Connected successfully! (Estimated FPS: {FPS_ESTIMATE:.2f})")
        return True
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

def send_image_uart(image: np.ndarray):
    """Send image via UART with header"""
    if serial_port is None or not serial_port.is_open:
        return
    
    try:
        # Ensure image is 128x128 grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Always normalize to 0-127 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume float is already 0-1 range
            image = (image * 127).astype(np.uint8)
        else:
            # Convert uint8 0-255 range to 0-127 range
            image = (image.astype(np.float32) / 255.0 * 127.0).astype(np.uint8)
        
        # Store sent image for display (scale back to 0-255 for visibility)
        display_img = (image.astype(np.float32) / 127.0 * 255.0).astype(np.uint8)
        try:
            sent_image_queue.put(display_img, block=False)
        except queue.Full:
            sent_image_queue.get()
            sent_image_queue.put(display_img)
        
        # Send header + image data
        with serial_lock:
            serial_port.write(UART_HEADER)
            serial_port.write(image.tobytes())
            serial_port.flush()
    except Exception as e:
        print(f"Error sending image: {e}")

def send_query():
    """Send query byte"""
    if serial_port is None or not serial_port.is_open:
        return
    
    try:
        with serial_lock:
            serial_port.write(QUERY_BYTE)
            serial_port.flush()
            time.sleep(0.1)
            response = serial_port.read(serial_port.in_waiting)
            print(f"Query response ({len(response)} bytes): {response.hex()}")
    except Exception as e:
        print(f"Error sending query: {e}")

# ==================== Thread: Sender ====================
def sender_thread():
    """Thread responsible for sending images from queue"""
    global running
    print("Sender thread started")
    
    while running:
        try:
            # Get image from queue with timeout
            packet = send_queue.get(timeout=0.1)
            send_image_uart(packet.image)
            send_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Sender thread error: {e}")
    
    print("Sender thread stopped")

# ==================== Thread: Receiver ====================
def receiver_thread():
    """Thread responsible for receiving CNN results with proper frame synchronization"""
    global running, webcam_active, result_mode
    print("Receiver thread started")
    
    # State machine: two states
    # 1. SEARCHING_FOR_HEADER: looking for header bytes using find()
    # 2. READING_FRAME: header found, accumulating frame data in chunks
    
    SEARCHING_FOR_HEADER = 0
    READING_FRAME = 1
    
    state = SEARCHING_FOR_HEADER
    search_buffer = bytearray()  # Buffer for header search
    frame_buffer = bytearray()   # Buffer for frame accumulation
    expected_frame_bytes = 0
    
    while running:
        if serial_port is None or not serial_port.is_open:
            time.sleep(0.1)
            continue
        
        try:
            if state == SEARCHING_FOR_HEADER:
                # Read chunk of data for header search
                if serial_port.in_waiting > 0:
                    with serial_lock:
                        chunk = serial_port.read(serial_port.in_waiting)
                    
                    search_buffer.extend(chunk)
                    
                    # Use find() to search for header (much faster than byte-by-byte)
                    header_pos = search_buffer.find(RESULT_HEADER)
                    
                    if header_pos != -1:
                        # Header found! Switch to reading frame
                        state = READING_FRAME
                        frame_buffer = bytearray()
                        
                        # Determine expected frame size based on mode
                        if result_mode == ResultType.CLASSIFICATION:
                            expected_frame_bytes = CLASS_COUNT * 1
                        elif result_mode == ResultType.SEGMENTATION:
                            expected_frame_bytes = SEGMENTATION_SIZE[0] * SEGMENTATION_SIZE[1] * 1
                        
                        # Any data after header becomes start of frame
                        remaining_after_header = search_buffer[header_pos + len(RESULT_HEADER):]
                        if remaining_after_header:
                            frame_buffer.extend(remaining_after_header)
                        
                        # Clear search buffer
                        search_buffer = bytearray()
                    else:
                        # Keep last (header_size - 1) bytes in case header is split across reads
                        if len(search_buffer) > len(RESULT_HEADER) - 1:
                            search_buffer = search_buffer[-(len(RESULT_HEADER) - 1):]
                else:
                    time.sleep(0.0001)
            
            elif state == READING_FRAME:
                # Calculate how many bytes we still need
                bytes_needed = expected_frame_bytes - len(frame_buffer)
                
                if bytes_needed > 0 and serial_port.in_waiting > 0:
                    with serial_lock:
                        # Read up to what we need, or what's available (chunk-based)
                        bytes_to_read = min(bytes_needed, serial_port.in_waiting)
                        chunk = serial_port.read(bytes_to_read)
                    
                    frame_buffer.extend(chunk)
                
                # Check if we have complete frame
                if len(frame_buffer) >= expected_frame_bytes:
                    # Queue raw frame data (no processing here!)
                    packet = RawFramePacket(
                        result_type=result_mode,
                        raw_data=bytearray(frame_buffer[:expected_frame_bytes]),  # Exact size
                        timestamp=time.time()
                    )
                    
                    try:
                        raw_frame_queue.put(packet, block=False)
                    except queue.Full:
                        raw_frame_queue.get()  # Remove old frame
                        raw_frame_queue.put(packet)
                    
                    # Any extra bytes go to search buffer for next header
                    if len(frame_buffer) > expected_frame_bytes:
                        search_buffer = bytearray(frame_buffer[expected_frame_bytes:])
                    else:
                        search_buffer = bytearray()
                    
                    # Go back to searching for next header
                    state = SEARCHING_FOR_HEADER
                    frame_buffer = bytearray()
                else:
                    # Still waiting for more frame data
                    if serial_port.in_waiting == 0:
                        time.sleep(0.0001)
            
        except Exception as e:
            print(f"Receiver thread error: {e}")
            # Reset state on error
            state = SEARCHING_FOR_HEADER
            search_buffer = bytearray()
            frame_buffer = bytearray()
            time.sleep(0.1)
    
    print("Receiver thread stopped")

# ==================== Thread: Webcam ====================
def webcam_thread():
    """Thread responsible for capturing webcam frames"""
    global running, webcam_active
    print("Webcam thread started")
    
    cap = None
    
    while running:
        with webcam_lock:
            should_capture = webcam_active
        
        if should_capture:
            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Failed to open webcam!")
                    with webcam_lock:
                        webcam_active = False
                    time.sleep(1)
                    continue
                print("Webcam opened")
            
            ret, frame = cap.read()
            if ret:
                # Store for display
                try:
                    display_queue.put(frame, block=False)
                except queue.Full:
                    display_queue.get()
                    display_queue.put(frame)
                
                # Prepare and queue for sending
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
                
                packet = ImagePacket(
                    image=resized,
                    timestamp=time.time()
                )
                
                try:
                    send_queue.put(packet, block=False)
                except queue.Full:
                    pass  # Skip frame if queue is full
            
            # Control frame rate
            time.sleep(1.0 / FPS_ESTIMATE)
        else:
            if cap is not None:
                cap.release()
                cap = None
                print("Webcam closed")
            time.sleep(0.1)
    
    if cap is not None:
        cap.release()
    print("Webcam thread stopped")

# ==================== Visualization ====================
def create_visualization(webcam_frame: Optional[np.ndarray], 
                        sent_image: Optional[np.ndarray],
                        result: Optional[ResultPacket]) -> np.ndarray:
    """Create visualization combining webcam, sent image, and results"""
    
    # Base canvas
    if webcam_frame is not None:
        canvas = webcam_frame.copy()
        height, width = canvas.shape[:2]
    else:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        height, width = 480, 640
    
    # Draw sent image in top-left corner
    if sent_image is not None:
        small_size = 120
        small_img = cv2.resize(sent_image, (small_size, small_size))
        if len(small_img.shape) == 2:
            small_img = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)
        
        # Add border
        cv2.rectangle(canvas, (5, 5), (5 + small_size, 5 + small_size), (0, 255, 0), 2)
        canvas[10:10+small_size, 10:10+small_size] = small_img
    
    # Draw results
    if result is not None:
        if result.result_type == ResultType.CLASSIFICATION:
            # Display classification results at top
            confidences = result.data
            top_class = np.argmax(confidences)
            top_conf = confidences[top_class]
            
            # Main result
            result_text = f"Digit: {top_class} ({top_conf*100:.1f}%)"
            cv2.rectangle(canvas, (width//2 - 150, 10), (width//2 + 150, 60), (0, 0, 0), -1)
            cv2.rectangle(canvas, (width//2 - 150, 10), (width//2 + 150, 60), (0, 255, 0), 2)
            cv2.putText(canvas, result_text, (width//2 - 140, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Top 3 predictions on the side
            top_3 = np.argsort(confidences)[-3:][::-1]
            for i, cls in enumerate(top_3):
                text = f"{cls}: {confidences[cls]*100:.1f}%"
                cv2.putText(canvas, text, (width - 150, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        elif result.result_type == ResultType.SEGMENTATION:
            # Overlay segmentation map
            seg_map = result.data
            
            # Resize to match frame
            seg_resized = cv2.resize(seg_map, (width, height))
            
            # Create red overlay where confidence > 0.5
            mask = (seg_resized > 0.1).astype(np.uint8) * 255
            red_overlay = np.zeros_like(canvas)
            red_overlay[:, :, 2] = mask  # Red channel
            
            # Blend with original
            canvas = cv2.addWeighted(canvas, 0.7, red_overlay, 0.3, 0)
            
            # Add segmentation stats
            seg_percent = np.mean(seg_map > 0.5) * 100
            text = f"Segmentation: {seg_percent:.1f}% detected"
            cv2.rectangle(canvas, (width//2 - 180, 10), (width//2 + 180, 50), (0, 0, 0), -1)
            cv2.putText(canvas, text, (width//2 - 170, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add status text
    status_text = "STREAMING" if webcam_active else "IDLE"
    status_color = (0, 255, 0) if webcam_active else (128, 128, 128)
    cv2.putText(canvas, status_text, (10, height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    return canvas

def display_thread():
    """Thread responsible for processing frames and visualization"""
    global running, running_v, webcam_active
    print("Display thread started")
    
    cv2.namedWindow('UART CNN Visualizer', cv2.WINDOW_NORMAL)
    
    current_frame = None
    current_sent = None
    current_result = None
    
    while running:
        
        if running_v:
            # Update webcam frame
            try:
                current_frame = display_queue.get(timeout=0.01)
            except queue.Empty:
                pass
            
            # Update sent image
            try:
                current_sent = sent_image_queue.get(timeout=0.01)
            except queue.Empty:
                pass
            
            # Process raw frames into results
            try:
                raw_packet = raw_frame_queue.get(timeout=0.01)
                
                # Process based on type
                if raw_packet.result_type == ResultType.CLASSIFICATION:
                    # Convert to uint8 array and normalize (0-127 -> 0-1)
                    confidences = np.frombuffer(bytes(raw_packet.raw_data), dtype=np.uint8).astype(np.float32) / 127.0
                    
                    current_result = ResultPacket(
                        result_type=ResultType.CLASSIFICATION,
                        data=confidences,
                        timestamp=raw_packet.timestamp
                    )
                
                elif raw_packet.result_type == ResultType.SEGMENTATION:
                    # Convert to 2D array and normalize (0-127 -> 0-1)
                    seg_data = np.frombuffer(bytes(raw_packet.raw_data), dtype=np.uint8).astype(np.float32) / 127.0
                    seg_map = seg_data.reshape(SEGMENTATION_SIZE)
                    
                    current_result = ResultPacket(
                        result_type=ResultType.SEGMENTATION,
                        data=seg_map,
                        timestamp=raw_packet.timestamp
                    )
                        
            except queue.Empty:
                pass
            
            # Create and show visualization
            viz = create_visualization(current_frame, current_sent, current_result)
            cv2.imshow('UART CNN Visualizer', viz)
            
            # Check for window close or ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty('UART CNN Visualizer', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        time.sleep(0.01)
    
    cv2.destroyAllWindows()
    print("Display thread stopped")

# ==================== Single Image Processing ====================
def process_single_image(filename: str):
    """Load and send a single image from img/ folder"""
    filepath = os.path.join("img", filename)
    
    global running_v, first_image
    running_v = False
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Load image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {filepath}")
        return
    
    # Resize and send
    resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    print(f"Sending image: {filename}")
    send_image_uart(resized)
    # sending 2 times, because we need it to actually get results from the CNN
    time.sleep(1)
    send_image_uart(resized)
    if first_image:
        first_image = False
        # sending a third time initially, because we need it to actually get results from the CNN
        time.sleep(1)
        send_image_uart(resized)
        
    
    # Wait for result
    print("Waiting for result...")
    start_time = time.time()
    timeout = 2.0
    
    while time.time() - start_time < timeout:
        try:
            # Process raw frame if available
            raw_packet = raw_frame_queue.get(timeout=0.1)
            raw_packet = raw_frame_queue.get(timeout=0.1)
            
            # Process the frame
            if raw_packet.result_type == ResultType.CLASSIFICATION:
                confidences = np.frombuffer(bytes(raw_packet.raw_data), dtype=np.uint8).astype(np.float32) / 127.0
                result = ResultPacket(
                    result_type=ResultType.CLASSIFICATION,
                    data=confidences,
                    timestamp=raw_packet.timestamp
                )
            elif raw_packet.result_type == ResultType.SEGMENTATION:
                seg_data = np.frombuffer(bytes(raw_packet.raw_data), dtype=np.uint8).astype(np.float32) / 127.0
                seg_map = seg_data.reshape(SEGMENTATION_SIZE)
                result = ResultPacket(
                    result_type=ResultType.SEGMENTATION,
                    data=seg_map,
                    timestamp=raw_packet.timestamp
                )
            else:
                continue
            
            if result.result_type == ResultType.CLASSIFICATION:
                # Print classification
                print("\nClassification Results:")
                for i, conf in enumerate(result.data):
                    print(f"  Digit {i}: {conf*100:.2f}%")
                top_class = np.argmax(result.data)
                print(f"\nPredicted: {top_class} (confidence: {result.data[top_class]*100:.2f}%)")
            
            elif result.result_type == ResultType.SEGMENTATION:
                running_v = True
                # Save segmentation (preserve original 0-127 range)
                seg_map = (result.data * 127).astype(np.uint8)
                #cv2.imwrite("out.png", seg_map)
                print(f"Segmentation saved to out.png (values 0-127)")
                #cv2.imwrite("in.png", resized)
                
                # Also show it (scale to 0-255 for visibility)
                seg_display = (result.data * 255).astype(np.uint8)
                cv2.imshow("Segmentation Result", seg_display)
                cv2.waitKey(2000)
                cv2.destroyWindow("Segmentation Result")
            
            return
        except queue.Empty:
            continue
    
    print("Timeout waiting for result")
    running_v = True

# ==================== Command Interface ====================
def command_interface():
    """Main command interface"""
    global running, webcam_active, result_mode
    
    print("\n" + "="*60)
    print("UART CNN Visualizer - Command Interface")
    print("="*60)
    print("Commands:")
    print("  ?               - Send test byte and print response")
    print("  c               - Start webcam stream")
    print("  s               - Stop webcam stream")
    print("  <filename>      - Send single image from img/ folder")
    print("  mode class      - Switch to classification mode")
    print("  mode seg        - Switch to segmentation mode")
    print("  e               - Exit")
    print("="*60 + "\n")
    
    while running:
        try:
            cmd = input("> ").strip()
            
            if cmd == "":
                continue
            elif cmd == "?":
                send_query()
            elif cmd == "c":
                with webcam_lock:
                    webcam_active = True
                print("Webcam stream started")
            elif cmd == "s":
                with webcam_lock:
                    webcam_active = False
                print("Webcam stream stopped")
            elif cmd == "e":
                print("Exiting...")
                running = False
                break
            elif cmd.startswith("mode "):
                mode = cmd.split()[1]
                if mode == "class":
                    result_mode = ResultType.CLASSIFICATION
                    print("Switched to CLASSIFICATION mode")
                elif mode == "seg":
                    result_mode = ResultType.SEGMENTATION
                    print("Switched to SEGMENTATION mode")
                else:
                    print("Unknown mode. Use 'class' or 'seg'")
            else:
                # Assume it's a filename
                process_single_image(cmd)
        
        except KeyboardInterrupt:
            print("\nExiting...")
            running = False
            break
        except Exception as e:
            print(f"Error: {e}")

# ==================== Main ====================
def main():
    """Main entry point"""
    global running
    
    print("Starting UART CNN Visualizer...")
    
    # Connect to UART
    if not connect_uart():
        print("Failed to establish UART connection. Exiting.")
        return
    
    # Create img directory if it doesn't exist
    os.makedirs("img", exist_ok=True)
    
    # Start threads
    threads = [
        threading.Thread(target=sender_thread, daemon=True),
        threading.Thread(target=receiver_thread, daemon=True),
        threading.Thread(target=webcam_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    # Give threads time to start
    time.sleep(0.5)
    
    try:
        # Run command interface
        command_interface()
    finally:
        # Cleanup
        print("Shutting down...")
        running = False
        
        # Wait for threads to finish
        for t in threads:
            t.join(timeout=2.0)
        
        # Close serial port
        if serial_port is not None and serial_port.is_open:
            serial_port.close()
        
        print("Goodbye!")

if __name__ == "__main__":
    main()
