import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\PRAJJWAL\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"

imgQ = cv2.imread('Query.png')
h, w, c = imgQ.shape
imgQ = cv2.resize(imgQ,(w//2,h//3))

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
imgKp1 = cv2.drawKeypoints(imgQ,kp1,None)

cv2.imshow("Keypoint",imgKp1)
cv2.imshow("OUTPUT",imgQ)

cv2.waitKey(0)
