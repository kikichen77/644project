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


## Function for real-time face detection and monitoring using a webcam. It compares detected faces with known encodings.
def real_time_monitoring(known_face_encodings):
    ## Load a pre-trained model for face detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    ## Initialize video capture from the webcam
    video_capture = cv2.VideoCapture(0)

    ## Counters and thresholds for face recognition
    registered_counter = 0
    unknown_counter = 0
    total_checks = 10
    alert_threshold = 2

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        ## Preprocess the frame and run it through the face detection model
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        ## Analyze each detection for face recognition
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face_encoding = face_recognition.face_encodings(frame, [(startY, endX, endY, startX)])
                if face_encoding:
                    match = compare_faces(known_face_encodings, face_encoding[0])

                    ## Update counters based on face recognition results
                    if match:
                        registered_counter += 1
                        unknown_counter = 0
                    else:
                        unknown_counter += 1
                        registered_counter = 0

                    ## Trigger an alert for unknown faces
                    if unknown_counter >= alert_threshold:
                        print("Alert: Unknown Face Detected!")
                        cv2.putText(frame, "Unknown", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    2)
                        unknown_counter = 0

                    ## Reset counters after a set number of checks
                    if registered_counter + unknown_counter >= total_checks:
                        registered_counter = 0
                        unknown_counter = 0

        ## Display the video frame with detected faces
        cv2.imshow('Video', frame)

        ## Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


def display_image_with_faces(image, face_locations):
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Resize the image to fit the screen
    resized_image = cv2.resize(image, (800, 600))
    cv2.imshow("Detected Faces", resized_image)
    cv2.waitKey(10)


def start_gui():
    def on_start_monitoring():
        filenames = select_images()
        if filenames:
            known_encodings = load_face_encodings(filenames)
            if known_encodings:
                # Load the image for face detection
                image = cv2.imread(filenames[0])
                face_locations = face_recognition.face_locations(image)
                display_image_with_faces(image, face_locations)
                real_time_monitoring(known_encodings)

    root = Tk()
    root.title("Face Recognition System")
    Label(root, text="Face Recognition System").pack()
    Button(root, text="Select Images", command=on_start_monitoring).pack()
    root.mainloop()


# Main function
if __name__ == "__main__":
    start_gui()
