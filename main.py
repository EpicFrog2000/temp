import cv2
import mediapipe as mp

# cascPath = "Scripts/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
#
# # Set video source to default webcam (at index 0)
# video_capture = cv2.VideoCapture(0)
#
# # Check if the video capture is opened correctly
# if not video_capture.isOpened():
#     print("Error: Camera not accessible")
#     exit()
#
# # Capture video
# while True:
#     ret, frame = video_capture.read()
#
#     # Check if frame is read correctly
#     if not ret:
#         print("Error: Frame not read properly")
#         break
#
#     try:
#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     except cv2.error as e:
#         print("Error during color conversion:", e)
#         break
#
#     # Detect faces
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )
#
#     # Draw rectangle around faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Display resulting frame
#     cv2.imshow('VIDEO', frame)
#
#     # Press Esc to exit
#     if cv2.waitKey(1) == 27:
#         break
#
# # Release video capture and close windows
# video_capture.release()
# cv2.destroyAllWindows()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    raise Exception("Error: Camera not accessible")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Frame not read properly")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        face_points = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * frame.shape[1]), int(
                        lm.y * frame.shape[0])

#OPCJONALNIE WYŚWIETLENIE PUNKTÓW
                    # cv2.putText(frame, f'{id}', (x, y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

                    face_points.append((id, x, y))

        print(face_points)

        cv2.imshow('VIDEO', frame)

        if cv2.waitKey(1) == 27:
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()
