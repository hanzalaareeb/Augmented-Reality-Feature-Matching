import cv2
import numpy as np


camera_matrix = np.array(
    [[517.47899846, 0, 339.53993095], [0, 516.14745585, 254.62370406], [0, 0, 1]],
    dtype=np.float32,
)
dist_coeffs = np.array(
    [-7.97239195e-02, 7.24817070e-01, 1.19007636e-03, 2.54971485e-03, -1.25542576e00],
    dtype=np.float32,
)

cap = cv2.VideoCapture(1)

# address = 'http://192.168.29.118:8080/video'
# cap.open(address)
imgTarget = cv2.imread("C:/Users/khanh/Desktop/IITM/DSA_ry/D06ss/Agu/Proj/Image01.jpg")

myVid = cv2.VideoCapture("chameleonspedup.mp4")
# detection of image if it is present then video will play
detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))
# arm detector
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)


# while loop to run the WebCam
while True:
    sucess, imgWebcam = cap.read()
    imgWebcam = cv2.undistort(imgWebcam, camera_matrix, dist_coeffs)
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)
    if des1 is None or des2 is None:
        print("Descriptors not found, skipping frame...")
        continue  # Skip this iteration of the loop
    if detection == False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    bf = cv2.BFMatcher()
    macthes = bf.knnMatch(des1, des2, k=2)
    good = []
    for match in macthes:
        # Check if there are at least two matches
        if len(match) >= 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good.append(m)

    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    if len(good) > 20:
        detection = True
        scrPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(scrPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

        imgwarp = cv2.warpPerspective(
            imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0])
        )

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgwarp, imgAug)

        # imgStacked = stackImages(([imgWebcam, imgVideo, imgTarget], [imgFeatures, imgwarp, imgAug]), 0.5)

    cv2.imshow("maskInv", imgAug)
    # cv2.imshow('imgwarp', imgwarp)
    # cv2.imshow('img2', img2)
    cv2.imshow("imgFeatures", imgFeatures)
    # cv2.imshow('ImgTarget', imgTarget)
    cv2.imshow("myVid", imgVideo)
    cv2.imshow("Webcam", imgWebcam)
    # cv2.imshow('imgStacked', imgStacked)
    cv2.waitKey(1)
    print("Frame count: ", frameCounter)
    frameCounter += 1
