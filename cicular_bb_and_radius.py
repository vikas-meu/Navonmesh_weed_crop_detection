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
cap = cv2.VideoCapture(3)

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
        mid_x = width // 2  # Calculate the center X-coordinate for the reference line
        cv2.line(local_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)  # Draw the green reference line

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

                    # Adjust the radius to be a smaller size relative to the detected bounding box
                    radius = int(min(w, h) / 3)  # Using min(w, h) divided by 3 to reduce the circle size

                    # Add the bounding box as [center_x, center_y, radius]
                    boxes.append([int(center_x), int(center_y), int(radius)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        if boxes:
            indices = cv2.dnn.NMSBoxes([[box[0] - box[2], box[1] - box[2], 2 * box[2], 2 * box[2]] for box in boxes], 
                                       confidences, 0.9, 0.9)
        else:
            indices = []

        # Detect if any plant crosses the reference line
        plant_crossed_line = False
        if len(indices) > 0:
            for i in indices.flatten():
                center_x, center_y, radius = boxes[i]
                # Draw a circular marker around the detected object
                cv2.circle(local_frame, (center_x, center_y), radius, (0, 255, 0), 2)

                # Display the Y-coordinate of the detected object center
                cv2.putText(local_frame, f"Y: {center_y}", (center_x - 20, center_y - radius - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Check if the center of the detected object crosses the vertical reference line
                if mid_x - 10 < center_x < mid_x + 10:
                    plant_crossed_line = True
                    cv2.putText(local_frame, "Crossed Line!", (center_x, center_y - radius - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
