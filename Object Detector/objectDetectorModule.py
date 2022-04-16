from distutils.command.config import config
import cv2


#img = cv2.imread('D:\\My Project Works\\OpenCV\\Object Detector\\lena.png')

classNames = []
classFile = 'D:\\My Project Works\\OpenCV\\Object Detector\\coco.names'

with open(classFile,'rt') as f:
    classNames  =f.read().rstrip('\n').split('\n')

configPath = 'D:\\My Project Works\\OpenCV\\Object Detector\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath  ="D:\\My Project Works\\OpenCV\\Object Detector\\frozen_inference_graph.pb"

net  = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

def getObject(img, thres, nms, draw = True, objects = []):
    classIds, confs, bbox  = net.detect(img, confThreshold = thres, nmsThreshold = nms)
    #print(classIds,bbox)

    if len(objects) == 0: objects = classNames 
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([box,className])
                if draw:
                    cv2.rectangle(img,box, color = (0,255,0),thickness=3)
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            
    return img,objectInfo

if __name__ == "__main__":
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