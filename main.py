import cv2
from nightlight import process

image = cv2.imread('test.png')

cv2.imshow("brightend Image", process(image))
