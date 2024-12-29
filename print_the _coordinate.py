import cv2
import numpy as np

# Load YOLO with optimized backend
net = cv2.dnn.readNet('/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/crop_weed_detection.weights',
                      '/home/pikas/Desktop/MYLIFE_MYPROJECT/crpo_and_weed_yolo/crop_weed.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Set OpenCV as the backend for better speed
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Or use cv2.dnn.DNN_TARGET_CUDA if you have GPU

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the video from a camera (change index if necessary)
cap = cv2.VideoCapture(3)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame or end of video")
        break

    # Resize frame to fit YOLO input size (416, 416)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pass the image through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process each detection
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Confidence threshold
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

    # Draw bounding boxes and labels only for weeds (class_id == 1)
    if len(indices) > 0:
        for i in indices.flatten():
            if class_ids[i] == 1:  # Only process weeds
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for weeds
                label = f"Weed: {x},{y}"  # Label the box with coordinates at the top
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Weed detected at: x={x}, y={y}, width={w}, height={h}")  # Print coordinates of the weeds

    # Show the current frame
    cv2.imshow("Frame", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
