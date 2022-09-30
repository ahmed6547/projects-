import numpy as np 
import cv2
import face_recognition
import os
from datetime import datetime

path= 'Nouveau dossier'
images =[]
classNames =[]
myList = os.listdir(path)
print(myList)
for i in myList:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])
print(classNames)
def findencodings(images):
    encodelist = []
    for img in images: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
        return encodelist
encodelistknown = findencodings(images)
print('encoding completed')


cap = cv2.VideoCapture(0)
while True : 
    succes, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)
    
    
    facecurframe = face_recognition.face_locations(imgS)
    encodescurframes = face_recognition.face_encodings(imgS, facecurframe)
    
    for encodeface,faceLoc in zip(encodescurframes, facecurframe):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis = face_recognition.face_distance(encodelistknown,encodeface)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1 ,(255,255,255),2)
        
             
             
             
             
             
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
        
    #test= face_recognition.load_image_file('Nouveau dossier/test.jpg')
#test=cv2.cvtColor(test,cv2.COLOR_RGB2BGR)
#joey= face_recognition.load_image_file('Nouveau dossier/joey.jpg')
#joey=cv2.cvtColor(joey,cv2.COLOR_RGB2BGR)