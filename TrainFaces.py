#Train Faces -- deep learning

import os
from PIL import Image
import numpy as np
import cv2
import pickle

#getting the directory of IDE
BASE_DIR= os.path.dirname(os.path.abspath(__file__))
#getting directory of where images are stored
img_dir= os.path.join(BASE_DIR, "train_faces")
#using Haar cascade classifier to detect faces
face_cascade= cv2.CascadeClassifier("F:\haarcascade_frontalface_alt2.xml")
#Recognition using LBPH Face Recogniser (can use eigen recogniser too)
recognizer= cv2.face.LBPHFaceRecognizer_create()

#Creating a dictionary for recognition
current_id= 0        #ID number
label_ids= {}        #Name
x_train= []
y_labels= []

for roots, dirs, files in os.walk(img_dir):
    for file in files:
        #reading images one by one
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path= os.path.join(roots, file)
            label= os.path.basename(os.path.dirname(path))
            #print(label, path)
            
            if not label in label_ids:
                label_ids[label]= current_id
                current_id+=1
                
            ids= label_ids[label]
            print(label_ids)
            pil_image= Image.open(path).convert("L")   #grayscale the image
            image_array= np.array(pil_image, "uint8")  #make that image in a numpy array
            #print(image_array)
            #Detect the face from the image
            faces= face_cascade.detectMultiScale(image_array , 1.5, 4)   
            for (x, y, w, h) in faces:
                roi= image_array[y:y+h , x: x+w]     #Region of interest
                x_train.append(roi)
                y_labels.append(ids)
                
#Storing the trained info into a file              
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)    
recognizer.train(x_train, np.array(y_labels)) 
recognizer.save("Trainer.yml")

    