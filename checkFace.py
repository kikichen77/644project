import cv2
import face_recognition

def real_time_monitoring(known_face_encodings, known_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 将视频帧转换为RGB，因为face_recognition使用RGB
        rgb_frame = frame[:, :, ::-1]

        # 检测帧中的所有脸部和脸部编码
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 检查每个面部是否与已知面部匹配
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # 绘制一个方框围绕脸部
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            if name == "Unknown":
                # 这里可以放置发送邮件的代码
                print("发送警报：检测到未知面孔")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
