import pafy
import cv2
import numpy as np

# Import xml haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Webcam
# cap = cv2.VideoCapture(0)
# Video
# cap = cv2.VideoCapture('./animeTest.mkv')

# Url Video
url = "https://www.youtube.com/watch?v=O7m7Rc6PwfE"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

count = 0

while True:
    # Read frame video
    _, img = cap.read()
    # Change color frame to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect in frame a face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Iterate detection images from algorithm
    for i, (x, y, w, h) in enumerate(faces):
        if i == 0:
            # Save firt image detect frame
            cv2.imwrite("img/frame%d.jpg" % count, img)
            count += 1
        # Paint rectangule in face detect
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show frames in external window
    cv2.imshow('img', img)

    # Detect 'Esc' key to close program
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()