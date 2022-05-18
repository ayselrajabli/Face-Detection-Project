import cv2
import time
import FaceDetectionModule as fdm

cap = cv2.VideoCapture("Videos/3.mp4")
pTime = 0
detector = fdm.FaceDetector()
while True:
    success, cam = cap.read()
    if not success:
        break
    cam, bboxs = detector.findFaces(cam)
    print(bboxs)

    cTime = time.time()
    fps = 1/(cTime - pTime) 
    pTime = cTime
    cv2.putText(cam, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,3,
                (255, 0, 0), 3)
    
    cv2.namedWindow("Video", 0)
    
    cv2.resizeWindow("Video", 1240, 760) 
    
    cv2.imshow("Video", cam)

    cv2.waitKey(10)