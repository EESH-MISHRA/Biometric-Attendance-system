import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cam = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    rate,frame = cam.read()
    frame,faces = detector.findFaceMesh(frame)
    cv2.imshow("frame",frame)
    key  = cv2.waitKey(1)
    if key == 27:
        break
