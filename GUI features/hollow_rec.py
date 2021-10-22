import numpy as np
import  cv2 as cv
import argparse

ref_point = []

def shape_selection(event, x, y, flags, param):
    global ref_point, crop
    if event == cv.EVENT_LBUTTONDOWN:
        ref_point = [(x,y)]
    elif event == cv.EVENT_LBUTTONUP:
        ref_point.append((x,y))
        cv.rectangle(img, ref_point[0], ref_point[1], (0,255,0), 1)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

path = './GUI features/images/' + args['image']
img = cv.imread(path)
clone = img.copy()
cv.namedWindow('image')
cv.setMouseCallback('image', shape_selection)

while True:
    cv.imshow('image', img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('c'): #if c then reset the window
        img = clone.copy()
    elif key == ord('q'):
        break

cv.destroyAllWindows()
