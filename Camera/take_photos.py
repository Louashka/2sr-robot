import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
import cv2

camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
i = 0
while 1:
    (grabbed, img) = camera.read()
    img = cv2.rotate(img, cv2.ROTATE_180)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press J to save image
        i += 1
        u = str(i)
        firename=str('Camera/Images/img'+u+'.jpg')
        cv2.imwrite(firename, img)
        print('writing：',firename)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Press Q to end
        break