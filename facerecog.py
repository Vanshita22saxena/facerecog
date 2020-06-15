import cv2

cap = cv2.VideoCapture('https://192.168.43.103:8080/video')

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
	status, photo = cap.read()
	#cphoto = photo[100:400, 200:500]
	#rphoto = cv2.resize(cphoto, (100,100))
	#photo[0:100 , 0:100] = rphoto
	face_cor = face_model.detectMultiScale(photo)

	if len(face_cor) == 0:
		pass
	else:
		x1 = face_cor[0][0]
		y1 = face_cor[0][1]
		x2 = x1 + face_cor[0][2]
		y2 = y1 + face_cor[0][3]

		photo = cv2.rectangle(photo, (x1,y1),(x2,y2), [0,255,0], 5)
		cv2.imshow('pic',photo)
		if cv2.waitKey(10) == 27:
			break

cv2.destroyAllWindows()
