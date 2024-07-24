import cv2
import numpy as np
import time

dist = np.array([ 3.58183122e-04, -2.20971875e-02, -2.46447770e-05,  1.46568391e-03, 6.40455235e-03])
newcameramtx = np.array([[591.4731274, 0.00000000e+00, 636.31026088],
                        [0.00000000e+00, 592.11854719, 354.22493702],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
mtx = np.array([[587.77695912, 0.00000000e+00, 636.67708427],
                        [0.00000000e+00, 587.49352402, 350.68017254],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = camera.read()
    cv2.imshow("original", frame)
    # frame = cv2.resize(frame, (1080, 720))
    h1, w1 = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Distortion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break