from tkinter import filedialog, Tk, Label, Button, Listbox
import face_recognition
import cv2
import numpy as np


def select_images():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames()
    root.destroy()
    return file_paths


def load_face_encodings(file_paths):
    all_encodings = []
    for file_path in file_paths:
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        all_encodings.extend(encodings)  # Add all encodings from each image
    return all_encodings

# Function to compare known face encodings with an unknown one
def compare_faces(known_face_encodings, unknown_face_encoding):
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
    return any(results)

# Function to detect and display faces using DNN and compare with selected images
def real_time_monitoring(known_face_encodings):
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_statuses = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face_encoding = face_recognition.face_encodings(frame, [(startY, endX, endY, startX)])
                if face_encoding:
                    match = compare_faces(known_face_encodings, face_encoding[0])
                    face_status = "Registered" if match else "Unknown"
                    face_statuses.append(face_status)

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 处理识别到的每个人脸的状态
        registered_faces = face_statuses.count("Registered")
        unknown_faces = face_statuses.count("Unknown")

        if unknown_faces > 2:
            print("Alert: Unknown Faces Detected!")
        elif registered_faces > 5:
            print("Multiple Registered Faces Detected!")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# GUI for selecting images and starting the monitoring
def start_gui():
    def on_start_monitoring():
        filenames = select_images()
        if filenames:
            known_encodings = load_face_encodings(filenames)
            if known_encodings:
                real_time_monitoring(known_encodings)

    root = Tk()
    root.title("Face Recognition System")
    Label(root, text="Face Recognition System").pack()
    Button(root, text="Select Images", command=on_start_monitoring).pack()
    root.mainloop()

# Main function
if __name__ == "__main__":
    start_gui()