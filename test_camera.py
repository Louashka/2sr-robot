import cv2

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("rotated by 180", frame)
    cv2.waitKey()

cap.release()