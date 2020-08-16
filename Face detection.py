import cv2
import numpy as np
face=cv2.CascadeClassifier('C:\OpenCV-Python-Series-master\src\cascades\data\haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=10)
    for(x,y,w,h) in faces:        #list of coordinates
        roi_gray=gray[y:y+h,x:x+w] 
        img_item="vism.jpg"
        cv2.imwrite(img_item,roi_gray)
        color=(255,0,0)
        stroke=2  #thickness
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    cv2.imshow('image',gray) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
