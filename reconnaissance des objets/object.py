import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)
#net =cv2.dnn.readNet("yolov3.weights","yolo.cfg")   #Load images 
classfile = 'coco.names'
classes =[]
with open(classfile, 'rt') as f :
    classes = [line.strip() for line in f.readlines()]
  
modelConfiguration = 'yolov3.cfg'
modelweights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelweights) 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_BACKEND_CUDA)
print (classes)
while True : 
    image = cv2.imread("image.jpg")
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    #height = image.shape(0)
    #width = image.shape(1)
    #channels =image.shape(2)
    
    blob = cv2.dnn.blobFromImage(image,1.0/255,(320,320),[0,0,0],1,crop=False)
    net.setInput(blob)
    
    layernames= net.getLayerNames()
    outputnames = [layernames[i - 1] for i in net.getUnconnectedOutLayers() ]
    print(net.getUnconnectedOutLayers())   
    output=[]
    for i in net.getUnconnectedOutLayers():
        output.append(layernames[i-1])
    #output = net.forward(outputnames)
    print(output)
    #for out in output:
       # for detection in out:
           # scores = detection[:]
            #classid = np.argmax(scores)
            #confidence = scores[classid]
            #if confidence > 0.5 : 
               # centerx =int(detection[0]*image.shape(0))



    cv2.imshow('Webcam',image)
    cv2.waitKey(1)

