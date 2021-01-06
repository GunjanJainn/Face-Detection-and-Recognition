# Fcae Detection with front camera

import cv2
import pickle


#Capture 
cap= cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier("F:\haarcascade_frontalface_alt2.xml")
#using the trained data 
recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels= {"person name": 1}
with open("labels.pickle", "rb") as f:
    og_labels= pickle.load(f)
    labels= {v:k for k, v in og_labels.items()}   #reversing the format in og_labels
    
while (cap.isOpened()):
    ret, frame= cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, cv2.CV_8U)
    #detect MultiScale works for cv2.CV_8U data type
    faces= face_cascade.detectMultiScale(gray, 1.2, 4)
    for (x, y, w, h) in faces:
        #Recognizing
        roi_gray= gray[ y: y+h, x: x+w]
        ids, conf= recognizer.predict(roi_gray)
        #Recognizing
        if conf> 45 and conf<75:
            print(ids)
            print(labels[ids])
        #Displaying the name of person
        cv2.putText(frame, labels[ids], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                    (255,0,0), 2, cv2.LINE_AA)
        #Detecting the region of interest
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow("Image", frame)
    if cv2.waitKey()== 32:   #Enter SpaceBar to exit the loop
        break
    
cv2.destroyAllWindows()
