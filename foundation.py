#IMPORTS
import numpy as np
import cv2
import pickle



#CASCADES
face_classifier = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")



#LOAD IN IDS
lables = {}
with open("labels.pickle", "rb") as f:
    original_labels = pickle.load(f)
    labels = {v:k for k, v in original_labels.items()}


#VIDEO CAPTURE
webcam = cv2.VideoCapture(0) #Index of default video device (webcam)

while(True):
    ret, frame = webcam.read() #Capturing frame by frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Changes frame to greyscale
    faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5) #Use the classifier with parameters
    for (x, y, w, h) in faces:
        #print(x, y, w, h) #Coordinates and deltas
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        final_id, conf = recognizer.predict(roi_gray)
        if conf >= 45: #and conf <= 85:
            print(final_id)
            print(labels[final_id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[final_id]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        img_item = "myFace.png"
        cv2.imwrite(img_item, roi_gray)
        color = (0, 0, 255) #Red box around face
        stroke = 3 #Thickness of box
        end_x_coord = x + w
        end_y_coord = y + h
        cv2.rectangle(frame, (x, y), (end_x_coord, end_y_coord), color, stroke)
    cv2.imshow("FRAME", frame) #Display each frame
    if cv2.waitKey(20) & 0xFF == ord("q"): #Built-in quit key
        break



#BREAK VIDEO CAPTURE
webcam.release() #Release the cature device
cv2.destroyAllWindows() #Close all frames
