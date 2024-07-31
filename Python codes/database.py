import os
import cv2
import imutils
import time
import pickle,re
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import copyreg as copy_reg 
import datetime
from pymongo import MongoClient
from random import randint
while True:
	if len(present) < 2:
	
		firebase_get_present()
	frame = vs.read()
	frame = imutils.resize(frame, width=1200)
	(h, w) = frame.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.95:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			if proba < 0.3:
				continue 
			text = "{}".format(name)
			
				
			y = startY - 10 if startY - 10 > 10 else startY + 10

			if text != "Unknown":
				if name in names:
					if name not in present:
						firebase_store(firebase_get(name))
				
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(255, 0, 0), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	fps.update()


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()