import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Initialize Tkinter window
root = tk.Tk()
root.title("AI Exam Proctoring System")
root.geometry("1000x750")

status_label = tk.Label(root, text="Waiting for detection...", font=("Helvetica", 14))
status_label.pack(pady=10)

# Increase camera resolution for a bigger frame
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Load Haarcascade for Face Detection (Fine-tuned)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

popup_shown = False

def apply_nms(faces, overlap_threshold=0.3):
    """ Apply Non-Maximum Suppression (NMS) to filter overlapping faces. """
    if len(faces) == 0:
        return []

    boxes = np.array(faces)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    
    area = (x2 - x1) * (y2 - y1)
    indices = np.argsort(y2)  # Sort by bottom-right y-coordinates
    selected = []

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        selected.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:-1]])
        yy1 = np.maximum(y1[i], y1[indices[:-1]])
        xx2 = np.minimum(x2[i], x2[indices[:-1]])
        yy2 = np.minimum(y2[i], y2[indices[:-1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[indices[:-1]]

        indices = indices[np.where(overlap < overlap_threshold)[0]]

    return [faces[i] for i in selected]

def detect_faces_and_objects():
    global popup_shown

    ret, frame = cap.read()
    if not ret:
        return

    # Ensure aspect ratio is maintained
    h, w, _ = frame.shape
    new_w = 1000  # Increased frame width
    new_h = int((new_w / w) * h)
    frame = cv2.resize(frame, (new_w, new_h))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improved Face Detection Parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(100, 100))

    # Apply Non-Maximum Suppression to filter overlapping faces
    faces = apply_nms(faces)

    # Use YOLO to detect objects (Optimized Parameters)
    results = model.predict(frame, conf=0.5, iou=0.45, verbose=False)

    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()
            label = model.names[class_id]  # Get object name

            if confidence > 0.5:  # Only consider high-confidence detections
                detected_objects.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    face_count = len(faces)

    # Improved Face Detection Logic
    if face_count > 1:
        if not popup_shown:
            messagebox.showwarning("Warning", "Multiple Faces Detected!")
            popup_shown = True
        status_label.config(text="Multiple Faces Detected!", fg="red")
    elif face_count == 0:
        status_label.config(text="No Face Detected", fg="orange")
        popup_shown = False
    else:
        status_label.config(text="Face Detected", fg="green")
        popup_shown = False

    # If a mobile is detected with high confidence
    if "cell phone" in detected_objects:
        messagebox.showwarning("Alert", "Mobile Phone Detected!")

    # Convert frame for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_img = Image.fromarray(frame)
    frame_img = ImageTk.PhotoImage(frame_img)
    frame_label.config(image=frame_img)
    frame_label.image = frame_img

    root.after(10, detect_faces_and_objects)

frame_label = tk.Label(root)
frame_label.pack()

detect_faces_and_objects()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
