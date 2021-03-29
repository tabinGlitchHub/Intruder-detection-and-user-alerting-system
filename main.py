import cv2
import methods as tools
import face_recognition as face_rec

thres = 0.60  # Threshold to detect object/ Confidence Score

# VideoCapture(0 for webcam, 1 for another first cam, 2 for another second cam and so on)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
haar_caascade = cv2.CascadeClassifier('haar_cascade.xml')

# assign neural net
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Change resolution
# tools.changeres(cap, 400, 400)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # if person detected, detect and recognize face
        if classIds[0] == 1:
            faces_rects = haar_caascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
            for (x, y, w, h) in faces_rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    cv2.imshow("Output", tools.resizewin(img))  # tools.resizewin(img) to resize window
    print(classIds, bbox)

    # waitkey(millisecond of delay between frames) and break loop if x is pressed
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
