import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PRAJJWAL\AppData\Local\Tesseract-OCR\tesseract.exe'

img = cv2.imread('plate.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

text = pytesseract.image_to_string(img)
print(text)

himg = img.shape
wimg = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    print(b)
    x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img,(x,(himg[0]-y)),(w,(himg[0]-h)),(0,0,255),1)
    cv2.putText(img,b[0],(x,himg[0]-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
    


cv2.imshow('RESULT',img)
cv2.waitKey(0)