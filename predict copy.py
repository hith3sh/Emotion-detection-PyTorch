import cv2
import torch
from FaceCNNModel import FaceCNN
import torchvision.transforms as transforms
import numpy as np



haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def detect_emotions(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in faces:
        #obtaining the face coordinates

        image = gray[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        #resizing the face image
        transform = transforms.Compose([
            transforms.Resize((48,48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        image = transform(image).unsqueeze(0) 
        labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

        
        feature = image
        pred = model(feature)
        prediction_label = labels[pred.argmax()]

        print("Predicted Output:", prediction_label)

        cv2.putText(frame, '% s' %(prediction_label), (x-10, y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))

    return frame