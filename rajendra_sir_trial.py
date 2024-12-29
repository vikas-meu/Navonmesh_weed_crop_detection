import cv2
import numpy as np
import threading
import tensorflow as tf
from pyfirmata import Arduino

board = Arduino('/dev/ttyACM0')
servo_pin = board.get_pin('d:9:s')

model = tf.keras.models.load_model('/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/weed_detection_rj_dt.h5')

cap = cv2.VideoCapture(3)

frame = None
stop_threads = False
lock = threading.Lock()

def capture_frames():
    global frame, stop_threads
    while not stop_threads:
        ret, new_frame = cap.read()
        if not ret:
            stop_threads = True
            break
        with lock:
            frame = new_frame

def process_efficientdet():
    global frame, stop_threads
    while not stop_threads:
        with lock:
            if frame is None:
                continue
            local_frame = frame.copy()

        input_frame = cv2.resize(local_frame, (512, 512)) / 255.0
        input_tensor = np.expand_dims(input_frame, axis=0).astype(np.float32)

        pred_class, pred_bbox = model.predict(input_tensor)

        plant_crossed_line = False
        height, width, _ = local_frame.shape
        mid_x = width // 2
        cv2.line(local_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)

        class_id = np.argmax(pred_class[0])
        startX, startY, endX, endY = [int(coord * width) for coord in pred_bbox[0]]

        box_center_x = startX + (endX - startX) // 2
        if box_center_x > mid_x - 10 and box_center_x < mid_x + 10:
            plant_crossed_line = True
            cv2.putText(local_frame, "Crossed Line!", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(local_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label = f"Class: {class_id}"
        cv2.putText(local_frame, label, (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        if plant_crossed_line:
            servo_pin.write(90)
        else:
            servo_pin.write(0)

        cv2.imshow(" Detection with Line", local_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

t1 = threading.Thread(target=capture_frames)
t2 = threading.Thread(target=process_efficientdet)

t1.start()
t2.start()

t1.join()
t2.join()

cap.release()
cv2.destroyAllWindows()

servo_pin.write(0)
board.exit()