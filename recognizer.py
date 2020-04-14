#IMPORTS
import os
import cv2
import numpy as np
from PIL import Image
import pickle



#MAIN REPOSITORIES
base = os.path.dirname(os.path.abspath(__file__))
image_repo = os.path.join(base, "images")



#CASCADES
face_classifier = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

#LOOP THROUGH IMAGES
y_labels = []
x_train = []
image_id = 0
label_ids = {} #ID Dictionary
for root, dirs, files, in os.walk(image_repo):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"): #Check for file extensions
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() #Format file names
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = image_id
                image_id += 1
            final_id = label_ids[label]
            pil_image = Image.open(path).convert("L") #Convert to grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_matrix = np.array(pil_image, "uint8") #Convert to Numpy array (matrix)
            #print(image_matrix)
            faces = face_classifier.detectMultiScale(image_matrix, scaleFactor = 1.5, minNeighbors=5) #Use the classifier with parameters
            for (x, y, w, h) in faces:
                roi = image_matrix[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(final_id)

print(x_train)
print(y_labels)


#DUMP LABLES
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
