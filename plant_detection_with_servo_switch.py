import cv2
import numpy as np
import threading
import time
from pyfirmata import Arduino, util

# Arduino setup with pyFirmata
board = Arduino('/dev/ttyACM0')  # Adjust to your Arduino port
servo_pin = board.get_pin('d:9:s')  # Pin 9 as servo pin (servo mode)

# Load YOLO
net = cv2.dnn.readNet('/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/crop_weed_detection.weights', 
                      '/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/crop_weed.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture (Camera 1)
cap = cv2.VideoCapture(1)

# Shared resources between threads
frame = None
stop_threads = False
lock = threading.Lock()

# Function to capture frames
def capture_frames():
    global frame, stop_threads
    while not stop_threads:
        ret, new_frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video")
            stop_threads = True
            break
        with lock:
            frame = new_frame

# Function to process YOLO
def process_yolo():
    global frame, stop_threads
    while not stop_threads:
        with lock:
            if frame is None:
                continue
            local_frame = frame.copy()  # Copy frame for processing

        # Prepare the frame for YOLO
        height, width, _ = local_frame.shape
        blob = cv2.dnn.blobFromImage(local_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # YOLO forward pass
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Detect if a plant (weed or crop) is found
        plant_detected = len(indices) > 0

        if plant_detected:
            # Move servo to 90 degrees if detection is found
            print("Plant detected! Moving servo to 90 degrees.")
            servo_pin.write(90)
        else:
            # Move servo back to 0 degrees
            print("No plant detected. Moving servo to 0 degrees.")
            servo_pin.write(0)

        # Draw bounding boxes if any plant detected
        if plant_detected:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(local_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Class: {class_ids[i]}"
                cv2.putText(local_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Show the processed frame
        cv2.imshow("YOLO Detection", local_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

# Start threads for frame capture and YOLO processing
t1 = threading.Thread(target=capture_frames)
t2 = threading.Thread(target=process_yolo)

t1.start()
t2.start()

# Wait for both threads to finish
t1.join()
t2.join()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Move servo to default position (0 degrees) before ending
servo_pin.write(0)
board.exit()
