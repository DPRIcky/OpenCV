from objectDetectorModule import *

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    success,img = cap.read()
    result,objectInfo = getObject(img,0.45,0.2,objects=['cell phone','person'])
    #print(objectInfo)
    cv2.imshow("OUTPUT",img)
    if cv2.waitKey(1) == ord('q'):
        break