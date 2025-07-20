import cv2
from vidstab import VidStab

# Initialize the stabilizer
stabilizer = VidStab()

# Open the webcam (input_path=0 is for default webcam)
cap = cv2.VideoCapture("http://192.168.29.118:8080/video")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break  # If there's an issue with capturing frames, exit the loop

    # Stabilize the current frame
    stabilized_frame = stabilizer.stabilize_frame(
        input_frame=frame, smoothing_window=30
    )

    # If stabilization result is available, show it in a window
    if stabilized_frame is not None:
        cv2.imshow("Stabilized Live Video", stabilized_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
