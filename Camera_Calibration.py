import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics = True):
    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'captured Images')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.png'))
    
    # initialize
    nRows = 9
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows*nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []
    
    # Initialize imgGray to None
    # imgGray = None

    
    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)
        
        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefind = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefind)
            
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefind, cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)
        """if cornersFound:
            cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersOrg, cornersFound)
            cv.imshow('Detected Chessboard Corners', imgBGR)
            cv.waitKey(0)  # Wait until you press a key to close the window
        else:
            print("Corners not found for image:", curImgPath)
            cv.imshow('Original Image', imgBGR)
            cv.waitKey(0)  # Display the original image to inspect manually"""

    cv.destroyAllWindows
    
    # Check if imgGray is assigned
    """if imgGray is None:
        print("No chessboard corners found in any image.")
        return"""
    
    # Calibrate Camera
    repError, camMatrix, distCoeff, revecs, tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print('Camera Matrix:\n', camMatrix)
    print('Reporj Error (pixels): {:.4f}'.format(repError))
    
    # savve Calibration parameters 
    
    curFolder = os.path.dirname(os.path.abspath(__file__))
    parmPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(parmPath,
             repError=repError,
             camMatrix=camMatrix,
             distCoeffs=distCoeff,
             revecs=revecs,
             tvecs=tvecs)
    return camMatrix, distCoeff

def removeDistortion(camMatrix, distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root, 'C:/Users/khanh/Desktop/IITM/DSA_ry/D06ss/Agu/captured Images/opencv_frame_8.png')
    img = cv.imread(imgPath)
    height, width =  img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 1, (width, height))
    imgUndist = cv.undistort(img, camMatrix, distCoeff, None, camMatrixNew)
    
    # Draw Lines to see Distortion change
    cv.line(img, (1769, 103), (1789, 922), (255, 255, 255), 2)
    cv.line(imgUndist, (1769, 103), (1789, 922), (255, 255, 255), 2)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()
    

def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix, distCoeff = calibrate()
    removeDistortion(camMatrix, distCoeff)

if __name__ == '__main__':
    # runCalibration()
    runRemoveDistortion()