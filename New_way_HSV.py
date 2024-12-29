import cv2
import numpy as np
import threading
import time
from pyfirmata import Arduino, util

board = Arduino('/dev/ttyACM0')   
servo_pin1 = board.get_pin('d:9:s')   
servo_pin2 = board.get_pin('d:10:s') 

cap = cv2.VideoCapture(3)

frame = None
stop_threads = False
lock = threading.Lock()

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

def process_hsv():
    global frame, stop_threads
    while not stop_threads:
        with lock:
            if frame is None:
                continue
            local_frame = frame.copy()

        height, width, _ = local_frame.shape
        mid_x = width // 2  # Calculate the midpoint of the frame width
        cv2.line(local_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)  # Draw the green reference line

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(local_frame, cv2.COLOR_BGR2HSV)

        # Define the HSV range for detecting green (assumed plant color)
        lower_bound = np.array([35, 100, 100])  # Adjust these values as needed
        upper_bound = np.array([85, 255, 255])

        # Create a mask for the green color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Find contours of the detected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plant_crossed_line = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small detections (adjust threshold as needed)
                x, y, w, h = cv2.boundingRect(contour)
                box_center_x = x + w // 2

                # Check if the plant crosses the line
                if mid_x - 10 < box_center_x < mid_x + 10:
                    plant_crossed_line = True
                    cv2.putText(local_frame, "Crossed Line!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Draw bounding boxes for detected objects
                cv2.rectangle(local_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Move servo based on whether a plant has crossed the line
        if plant_crossed_line:
            print("Plant detected crossing the line! Moving servo to 90 degrees.")
            servo_pin1.write(90)
            servo_pin2.write(90)
        else:
            print("No plant crossing the line. Moving servo to 0 degrees.")
            servo_pin1.write(0)
            servo_pin2.write(0)

        # Show the processed frame
        cv2.imshow("HSV Detection with Line", local_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

# Start threads for frame capture and HSV processing
t1 = threading.Thread(target=capture_frames)
t2 = threading.Thread(target=process_hsv)

t1.start()
t2.start()

# Wait for both threads to finish
t1.join()
t2.join()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Move servo to default position (0 degrees) before ending
servo_pin1.write(0)
servo_pin2.write(0)
board.exit()
