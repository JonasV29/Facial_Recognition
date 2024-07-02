import cv2 as cv #importando OpenCV library


video = cv.VideoCapture(0)  # abre a câmera

face_detect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID:") # entrar com seu ID
#id = int(id)
count = 0

while True:
    ret,frame=video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:  # criação das fotos capturadas do dados de entrada
        count = count + 1
        cv.imwrite('dataset/User.' +str(id) + '.' +str(count)+'.jpg', gray[y:y+h, x:x+w])
        cv.rectangle(frame, (x,y),(x+w,y+h),(50,50,255),1) # criação do quadrado onde contorna o rosto


    cv.imshow("Frame",frame)

    K = cv.waitKey(1)

    if count > 100: # conta quanto que é a captura
        break

video.release()
cv.destroyAllWindows()
print("Dataset colletion Done....")