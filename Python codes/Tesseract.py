import cv2
import pytesseract
import pyttsx3
import RPi.GPIO as GPIO
import time

BUTTON_PIN = 29
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        return "captured_image.jpg"
    else:
        return None


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)  
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, thresh)
    return processed_image_path


def extract_text(image_path):
    processed_image_path = preprocess_image(image_path)
    img = cv2.imread(processed_image_path)
    text = pytesseract.image_to_string(img)
    return text

def read_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    engine.say(text)
    engine.runAndWait()

try:
    while True:
        button_state = GPIO.input(BUTTON_PIN)
        if button_state == GPIO.LOW:  
            image_path = capture_image()
            if image_path:
                text = extract_text(image_path)
                print("Extracted Text:", text)
                read_text(text)
            time.sleep(1)  
finally:
    GPIO.cleanup()
