import os
import cv2 as cv


# Discription: to check if camera is running
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Unable to read camera fee")
else:
    print("YaaaHoooo")

# Creating a collection of images
data_collection = "captured Images"
os.makedirs(data_collection, exist_ok=True)

imageCounter = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv.imshow("webcam", frame)
    k = cv.waitKey(1)
    if k % 256 == 27:  # 'Esc' key to terminate the session
        print("Escape hit, closing..")
        break

    elif k % 256 == ord("s"):  # 's' key to save the frame
        img_name = os.path.join(
            data_collection, "opencv_frame_{}.png".format(imageCounter)
        )
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        imageCounter += 1

cap.release()
cv.destroyAllWindows
