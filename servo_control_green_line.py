import cv2
import numpy as np
import threading
import time
from pyfirmata import Arduino, util

board = Arduino('/dev/ttyACM0')   
servo_pin1 = board.get_pin('d:9:s')   
servo_pin2 = board.get_pin('d:10:s') 
 
net = cv2.dnn.readNet('/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/crop_weed_detection.weights', 
                      '/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/crop_weed.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

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

 
def process_yolo():
    global frame, stop_threads
    while not stop_threads:
        with lock:
            if frame is None:
                continue
            local_frame = frame.copy()   

       
        height, width, _ = local_frame.shape
        mid_x = width // 2  
        cv2.line(local_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)  # Draw the green reference line

        blob = cv2.dnn.blobFromImage(local_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
     
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
 
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:   
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

         
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.7)

        
        plant_crossed_line = False
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                box_center_x = x + w // 2

                
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
            servo_pin1.write(0)
            servo_pin2.write(0)
        else:
            print("No plant crossing the line. Moving servo to 0 degrees.")
            servo_pin1.write(94)
            servo_pin2.write(94)

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
servo_pin1.write(94)
servo_pin2.write(94)
board.exit()
