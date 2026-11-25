
import cv2
import os

video = cv2.VideoCapture("dataset/people.mp4")

os.makedirs("frames", exist_ok=True)

count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imwrite(f"frames/frame_{count}.jpg", frame)

    count += 1

video.release()

print("Total frames extracted:", count)

import cv2
import os

os.makedirs("processed", exist_ok=True)

for filename in os.listdir("frames"):

    img = cv2.imread("frames/" + filename)

    # Resize image
    resized = cv2.resize(img, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("processed/" + filename, gray)

print("Image preprocessing completed!")
