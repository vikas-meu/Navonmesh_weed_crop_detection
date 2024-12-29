import cv2
import numpy as np
import threading
import time
from pyfirmata import Arduino, util
from tensorflow.keras.models import load_model

# Arduino setup with pyFirmata
board = Arduino('/dev/ttyACM0')  # Adjust to your Arduino port
servo_pin = board.get_pin('d:9:s')  # Pin 9 as servo pin (servo mode)

# Load the Keras model
model = load_model('/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/weed_detection_rj_dt.h5')

# You may need to adjust the input shape and preprocessing depending on your specific model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (416, 416))  # Assuming the model expects 416x416 input
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1] range if required
    return np.expand_dims(normalized_frame, axis=0)

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

# Function to process YOLO using Keras model
def process_yolo():
    global frame, stop_threads
    while not stop_threads:
        with lock:
            if frame is None:
                continue
            local_frame = frame.copy()  # Copy frame for processing

        # Prepare the frame for the model
        height, width, _ = local_frame.shape
        mid_x = width // 2  # Calculate the center X-coordinate for the reference line
        cv2.line(local_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)  # Draw the green reference line

        # Preprocess frame and run inference
        processed_frame = preprocess_frame(local_frame)
        predictions = model.predict(processed_frame)[0]  # Get the first batch result

        # Assume your predictions array contains information like [center_x, center_y, width, height, class_id, confidence]
        class_ids, confidences, boxes = [], [], []

        for pred in predictions:
            class_id = int(pred[4])  # Example of how you might retrieve class_id
            confidence = pred[5]     # Example of how you might retrieve confidence

            if confidence > 0.5:  # Confidence threshold
                center_x = int(pred[0] * width)
                center_y = int(pred[1] * height)
                w = int(pred[2] * width)
                h = int(pred[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Apply non-maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)

        # Detect if any plant crosses the reference line
        plant_crossed_line = False
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                box_center_x = x + w // 2

                # Check if the center of the detected plant crosses the vertical reference line
                if box_center_x > mid_x - 10 and box_center_x < mid_x + 10:
                    plant_crossed_line = True
                    cv2.putText(local_frame, "Crossed Line!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Draw bounding boxes for detected objects
                cv2.rectangle(local_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Class: {class_ids[i]}"
                cv2.putText(local_frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        # Move servo based on whether a plant has crossed the line
        if plant_crossed_line:
            print("Plant detected crossing the line! Moving servo to 90 degrees.")
            servo_pin.write(90)
        else:
            print("No plant crossing the line. Moving servo to 0 degrees.")
            servo_pin.write(0)

        # Show the processed frame
        cv2.imshow("YOLO Detection with Line", local_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

# Start threads for frame capture and YOLO processing
t1 = threading.Thread(target=capture_frames)
t2 = threading.Thread(target(process_yolo))

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
