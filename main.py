import cv2

# Create face cascade
cascPath = "/Users/janekkorczynski/Desktop/PaidWork/face_recognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Set video source to default webcam (at index 0)
video_capture = cv2.VideoCapture(0)

# Check if the video capture is opened correctly
if not video_capture.isOpened():
    print("Error: Camera not accessible")
    exit()

# Capture video
while True:
    ret, frame = video_capture.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Frame not read properly")
        break

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print("Error during color conversion:", e)
        break

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display resulting frame
    cv2.imshow('VIDEO', frame)
    
    # Press Esc to exit
    if cv2.waitKey(1) == 27:
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
