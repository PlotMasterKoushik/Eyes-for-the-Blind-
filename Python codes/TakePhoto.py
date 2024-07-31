import cv2
import sys
import time 

video_capture = cv2.VideoCapture(0)
image_count = 0 
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        pass

    ret, image = video_capture.read()
   
    if image_count <= 16: 
        time.sleep(0.5) 
        cv2.imwrite('dataset/{}-Aaryan.jpg'.format(image_count), image)
    if image_count >= 16:
        print("Done")
        break
    
    image_count += 1
    cv2.imshow('Video', image) 


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
video_capture.release()
cv2.destroyAllWindows()