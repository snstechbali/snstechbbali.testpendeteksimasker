import cv2 #memanggil fungsi

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Memuat Data pendeteksian wajah
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#Memuat Data pendeteksian mata
nose_cascade = cv2.CascadeClassifier('Nariz_Hidung.xml')#Memuat Data pendeteksian hidung

cap = cv2.VideoCapture(0)#Agar dapat menampilkan video capture

while True:
    _, frame = cap.read()#Agar terbentuk frame jika wajah sudah terdeteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Mengkonversikan gambar agar menjadi skala
    
#Mendeteksi Wajah
    face = face_cascade.detectMultiScale(gray, 1.3, 5)#Mendeteksi wajah menggunakan detectMultiScale
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (225, 0, 0), 3)#Memberi warna pada frame rectangle
        cv2.putText (frame, 'face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)#Memberi warna pada sebuah Text

        roi_gray = gray[y:y+h, x:x+w]#pelabelan data
        roi_color = frame[y:y+h, x:x+w]#pelabelan data

#Mendeteksi Mata
        eye = eye_cascade.detectMultiScale(roi_gray)#Mendeteksi mata menggunakan detectMultiScale
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0),2)#Memberi warna pada frame rectangle ketika mata sudah terdeteksi
            cv2.putText (frame,'eye', (x + ex,y +ey), 1, 2,(0, 255, 0),2)#Memberi warna pada sebuat text

#Mendeteksi Hidung 
        nose = nose_cascade.detectMultiScale(gray, 1.18, 35)#Mendeteksi hidung menggunakan detectMultiScale
        for (sx,sy,sw,sh) in nose:
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0),2)#Memberi warna pada frame rectangle ketika hidung sudah terdeteksi
            cv2.putText (frame,'nose', (x + sx,y +sy), 1, 3,(255, 0, 0),2)#Memberi warna pada sebuah text

    cv2.putText (frame, 'jumlah wajah : ' + str(len(face)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),2)#Menambahkan text untuk menampilkan jumlah objek
    cv2. imshow('Face', frame)#Dihitung ketika wajah terdeteksi 

    if cv2.waitKey(30) & 0xff == ord('q'):#Menunggu objek
        break
cap.release()
cv2.destroyAllWindows()#Menutup semua jendela

        
