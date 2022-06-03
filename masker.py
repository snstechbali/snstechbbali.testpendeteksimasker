import cv2#Memanggil fungsi
import os#memanggil sebagian fungsi pada OpenCv


mask_on = False #Memberikan nilai yaitu False


#Memuat data pendeteksian wajah, mata, dan hidung
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('Nariz_Hidung.xml')

cap = cv2.VideoCapture(0)#Agar dapat menampilkan video capture


while True:
    _, frame = cap.read()#Agar terbentuk frame jika wajah sudah terdeteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Mengkonversikan gambar agar menjadi skala

    
#Mendeteksi Wajah
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        if mask_on:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (225, 0, 0), 3)
            cv2.putText (frame, 'Mask on', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)
            os.system("start alarm.M4A")#Memanggil fungsi agar dapat menambahkan suara
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText (frame, 'Mask off', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)
            
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

#Mendeteksi Mata
        eye = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0),2)
            cv2.putText (frame,'eye', (x + ex,y +ey), 1, 2,(0, 255, 0),2)

#Mendeteksi Hidung 
        nose = nose_cascade.detectMultiScale(gray, 1.18, 35)
        for (sx,sy,sw,sh) in nose:
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0),2)
            cv2.putText (frame,'nose', (x + sx,y +sy), 1, 3,(255, 0, 0),2)

#Respon ketika hidung terdeteksi dan tidak terdeteksi
        if len(nose)>0:
            mask_on = False
        else:
            mask_on = True

    cv2.putText (frame, 'jumlah wajah : ' + str(len(face)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),2)
    cv2. imshow('Face', frame)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

        
