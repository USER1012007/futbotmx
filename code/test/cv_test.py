"""
    Este archivo unicamente incluye pruebas con la libreria cv2, directamente desde la documentacion
"""

import cv2 as cv
import sys

# Load an image
img = cv.imread('image.jpg')

# Check if image was loaded successfully
if img is None:
    sys.exit("Could not read the image.")

# Display the image
cv.imshow("Display window", img)
k = cv.waitKey(0)

# Save if 's' key is pressed
if k == ord("s"):
    cv.imwrite("output.png", img)

#####################################################

import cv2 as cv

# Read the image
img = cv.imread('image.jpg')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv.Canny(blurred, 50, 150)

# Display all results
cv.imshow('Original', img)
cv.imshow('Grayscale', gray)
cv.imshow('Edges', edges)

cv.waitKey(0)
cv.destroyAllWindows()

#####################################################

# Read image
img = cv.imread('input.jpg')

# Save image
cv.imwrite('output.jpg', img)

# Read video
cap = cv.VideoCapture('video.mp4')

# Write video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#####################################################

# Resize

img = cv.imread('input.jpg')
resized = cv.resize(img, (640, 480))

# Rotate
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, 45, 1.0)
rotated = cv.warpAffine(img, M, (w, h))

# Flip
flipped = cv.flip(img, 1)  # 1 = horizontal, 0 = vertical, -1 = both
cv.imshow("Flipped image", flipped)
cv.waitKey(0)

#####################################################

# Draw line
cv.line(img, (0, 0), (100, 100), (255, 0, 0), 2)

# Draw circle
cv.circle(img, (50, 50), 25, (0, 255, 0), -1)

# Draw rectangle
cv.rectangle(img, (10, 10), (100, 100), (0, 0, 255), 2)

# Put text
cv.putText(img, 'Hello OpenCV', (10, 30), 
           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv.imwrite('output.jpg', img)
