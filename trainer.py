import os
import cv2 as cv
import numpy as np

friends = ['aditya', 'sidhant', 'tabin']

DIR = r'./Images'

haar_cascade = cv.CascadeClassifier('haar_cascade.xml')

features = []
labels = []


def train():
    for friend in friends:
        path = os.path.join(DIR, friend)
        label = friends.index(friend)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            for (x, y, h, w) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


train()
print(f'features:{len(features)}')
print(f'labels:{len(labels)}')
print('-------Training Done-------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
