import cv2
import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
from ipywidgets import interact, Layout

width_img = 1280
height_img = 960

def preprocess(fr, v1):
    mask = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (5, 5), 2)
    mask = cv2.Canny(mask, v1[0], v1[1], apertureSize=5)
    cv2.imshow('mask Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask

def contours(fr_m, fr):
    biggest = np.array([])
    max_area = 0
    contours, _ = cv2.findContours(fr_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    return my_points_new

def get_warp(img, approx):
    biggest = reorder(approx)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width_img, height_img))
    return img_output

path = 'images/img4.jpg'
img = cv2.imread(path)

img = cv2.resize(img, (width_img, height_img))

frame_masked = preprocess(img, [255, 255])

frame_contours = contours(frame_masked, img)

if frame_contours.size != 0:
    img_warped = get_warp(img, frame_contours)
    cv2.imshow('Warped Image', img_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
