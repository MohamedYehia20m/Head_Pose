import time
import helper

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from numpy import random

#cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    #success, img = cap.read()
    import os

    os.chdir(r"F:\CV_material")
    img = cv2.imread("ana4.jpg")
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        #randomPoint = face[400]
        #cv2.circle(img, randomPoint , 5, (255, 0, 255), cv2.FILLED)

        randomPoint2 = face[random.randint(1, 468)]
        cv2.circle(img, randomPoint2, 5, (255, 0, 255), cv2.FILLED)

        time.sleep(1)



        # Drawing
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        # print(w)
        W = 6.3

        # # Finding the Focal Length
        d = 50
        f = (w*d)/W
        print(f)

        # Finding distance
        #f = 840
        #d = (W * f) / w
        #print(d)

        cvzone.putTextRect(img, f'{helper.is_closer(pointRight,pointLeft,randomPoint2)}',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()