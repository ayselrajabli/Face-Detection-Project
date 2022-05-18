from tkinter.tix import Tree
import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
    
    def findFaces(self, cam, draw = True):
        camRGB = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(camRGB)
        #print(results)
        bboxs = []

        if results.detections:
            for id, detection in enumerate(results.detections):
                #mpDraw.draw_detection(cam, detection)
                # print(id, detection)
                # print(detection.score)
                print(detection.location_data.relative_bounding_box.xmin)
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = cam.shape
                bbox =int(bboxC.xmin*w), int(bboxC.ymin*h), \
                    int(bboxC.width*w), int(bboxC.height*h)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    cam = self.fancyDraw(cam, bbox)
                    cv2.putText(cam, f'{int(detection.score[0]*100)}%',
                                (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,3,
                                (255, 0, 255), 3)
        return cam, bboxs
    
    def fancyDraw(self, cam, bbox, l = 30, t = 5, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(cam, bbox, (255, 0, 255), rt)
        #Up Left
        cv2.line(cam, (x,y), (x+l, y), (255, 0, 255),t)
        cv2.line(cam,  (x,y), (x, y+l), (255, 0, 255),t)
        #Up Right
        cv2.line(cam, (x1,y), (x1-l, y), (255, 0, 255),t)
        cv2.line(cam,  (x1,y), (x1, y+l), (255, 0, 255),t)
        #Down Left
        cv2.line(cam, (x,y1), (x+l, y1), (255, 0, 255),t)
        cv2.line(cam,  (x,y1), (x, y1-l), (255, 0, 255),t) 
        #Left
        cv2.line(cam, (x1,y1), (x1-l, y1), (255, 0, 255),t)
        cv2.line(cam,  (x1,y1), (x1, y1-l), (255, 0, 255),t)       
        return cam
            


def main():
    cap = cv2.VideoCapture("Videos/3.mp4")
    pTime = 0
    detector = FaceDetector()
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


if __name__ == "__main__":
    main()

