import cv2 as cv #importing OpenCV library


video = cv.VideoCapture(0)

face_detect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = [ "","Jonas Vitorio", "Jason"]


while True:
    ret,frame=video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf>50:
            cv.putText(frame,name_list[serial],(x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,255),2)
            cv.rectangle(frame, (x,y),(x+w,y+h),(50,50,255),1)
        else:
            cv.putText(frame,"Unknown",(x,y-10), cv.FONT_HERSHEY_SIMPLEX,0.8, (50,50,255),2)
            cv.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
   
    
    cv.imshow("frame", frame)

    K = cv.waitKey(1)

    if  K == ord("q"):
        break

video.release()
cv.destroyAllWindows()
print("Facial Recognition Done....")