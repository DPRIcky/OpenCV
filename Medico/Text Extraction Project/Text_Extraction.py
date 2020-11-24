import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\PRAJJWAL\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"

per = 25

roi = [[(130, 332), (354, 384), 'Text', 'Prescription number'],
       [(887, 332), (1092, 382), 'Text', 'DATE'],
       [(149, 504), (554, 562), 'Text', "Patient's Name"],
       [(149, 647), (644, 814), 'Text', "Patient's Address"],
       [(792, 647), (1137, 717), 'Text', 'Phone Number'],
       [(149, 884), (404, 947), 'Text', 'Gender'],
       [(794, 892), (1044, 947), 'Text', 'Age'],
       [(144, 1034), (407, 1094), 'Text', 'Weight'],
       [(152, 1229), (589, 1552), 'Text', 'Medicine']]



imgQ = cv2.imread('Query.png')
h, w, c = imgQ.shape
#imgQ = cv2.resize(imgQ,(w//2,h//3))

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
imgKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'UserForms'
mypiclist = os.listdir(path)
print(mypiclist)
for j,y in enumerate(mypiclist):
    img = cv2.imread(path + '/' + y)
    #img  = cv2.resize(img,(w//2,h//3))
    # cv2.imshow(y,img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key = lambda x:x.distance)
    good = matches[:int(len(matches)*per/100)]
    imgmatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags = 2)
    # imgmatch  = cv2.resize(imgmatch,(w//2,h//3))
    # cv2.imshow(y,imgmatch)
    
    srcpoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstpoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, _ = cv2.findHomography(srcpoints,dstpoints,cv2.RANSAC,5.0)
    imgscan = cv2.warpPerspective(img,M,(w,h))
    #imgscan  = cv2.resize(imgscan,(w//2,h//3))
    #cv2.imshow(y,imgscan)
    
    imgshow = imgscan.copy()
    imgmask = np.zeros_like(imgshow)
    
    myData = []
    
    for x,r  in enumerate(roi):
        
        cv2.rectangle(imgmask,(r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgshow = cv2.addWeighted(imgshow,0.99,imgmask,0.1,0)
        
        imgCrop= imgscan[r[0][1]:r[1][1],r[0][0]:r[1][0]]
        cv2.imshow(str(x),imgCrop)
        
        if r[2] == 'Text':
            print('{}: {}'.format(r[3],pytesseract.image_to_string(imgCrop)))
            myData.append(pytesseract.image_to_string(imgCrop))
    
    with open('DataOutput.csv','a+') as f:
        for data in myData:
            f.write(str(data)+',')
        f.write('\n')
    
    imgshow  = cv2.resize(imgshow,(w//2,h//3))
    cv2.imshow(y+"2",imgshow)

    

#cv2.imshow("Keypoint",imgKp1)
cv2.imshow("OUTPUT",imgQ)

cv2.waitKey(0)
