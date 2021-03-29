import cv2 as cv

# change resoltuion of input frames
def changeres(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# resize window
def resizewin(cap, scale=0.5):
    width = int(cap.shape[0] * scale)
    height = int(cap.shape[1] * scale)

    dimension = (height, width)
    return cv.resize(cap, dimension, interpolation=cv.INTER_AREA)
