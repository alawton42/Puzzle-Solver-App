import cv2
import numpy as np

"""
Test code for OpenCV.
Primarily background removal at the moment
TODO:
    - Experiment with how to identify piece and separate from the rest of the image
    - Remove after project completion
"""

# img = cv2.imread('testimage.png')
# img2 = numpy.array(img)
#
# print(img[30])
# cv2.imwrite("test_img_row.png", img)

# src = cv2.imread('puzzleTest.png', cv2.IMREAD_UNCHANGED)
#
# # bgr = src[:, :, :3]  # Channels 0..2
# # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#
# # Some sort of processing...
# oop = cv2.Canny(src, 100, 200)
#
# # bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# # alpha = src[:, :, 3]  # Channel 3
# black_mask = np.all(oop == 0, axis=-1)
# alpha = np.uint8(np.logical_not(black_mask)) * 255
# result = np.dstack((oop, alpha))  # Add the alpha channel
#
# cv2.imwrite('test_img_row.png', oop)




# import numpy as np
# import cv2
# color = cv2.imread("puzzleTest.png", 1)
# black_mask = np.all(color == 0, axis=-1)
# alpha = np.uint8(np.logical_not(black_mask)) * 255
# bgra = np.dstack([color, alpha])
# print(type(bgra))
# cv2.imwrite("aaa.png", bgra)


# #== Parameters
# BLUR = 21
# CANNY_THRESH_1 = 10
# CANNY_THRESH_2 = 200
# MASK_DILATE_ITER = 10
# MASK_ERODE_ITER = 10
# MASK_COLOR = (0.0,0.0,1.0) # In BGR format
#
#
# #-- Read image
# img = cv2.imread('puzzleTest.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# #-- Edge detection
# edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
# edges = cv2.dilate(edges, None)
# edges = cv2.erode(edges, None)
#
# #-- Find contours in edges, sort by area
# # contour_info = []
# # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# # for c in contours:
# #     contour_info.append((
# #         c,
# #         cv2.isContourConvex(c),
# #         cv2.contourArea(c),
# #     ))
# # contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
# # max_contour = contour_info[0]
# #
# # print(max_contour[0])
#
# #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# # Mask is black, polygon is white
# mask = np.zeros(edges.shape)
# # cv2.fillConvexPoly(mask, max_contour[0], (255))
# tmp =  cv2.floodFill(gray, mask, (0, 0), 255)
#
# #-- Smooth mask, then blur it
# mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
# mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
# mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
# mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
#
# #-- Blend masked img into MASK_COLOR background
# mask_stack  = mask_stack.astype('float32') / 255.0
# img         = img.astype('float32') / 255.0
# masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
# masked = (masked * 255).astype('uint8')
#
# # cv2.imshow('img', mask)                                   # Display
# # cv2.waitKey()
# cv2.imwrite("aaa.png", tmp)



# # Read image
# # og = cv2.imread("puzzleTest.png")
# # im_in = cv2.imread("puzzleTest.png", cv2.IMREAD_GRAYSCALE)
#
# og = cv2.imread("IMG_0073.jpg")
# im_in = cv2.imread("IMG_0073.jpg", cv2.IMREAD_GRAYSCALE)
#
# # Threshold.
# # Set values equal to or above 220 to 0.
# # Set values below 220 to 255.
#
# # im_in = cv2.Canny(og, 200, 300)
# th, im_th = cv2.threshold(im_in, 130, 255, cv2.THRESH_BINARY_INV)
#
#
# # Copy the thresholded image.
# im_floodfill = im_th.copy()
#
#
# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
# h, w = im_th.shape[:2]
# mask = np.zeros((h + 2, w + 2), np.uint8)
#
# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0, 0), 0)
#
# # Invert floodfilled image
# # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
# # Combine the two images to get the foreground.
# # im_out = im_th | im_floodfill_inv
#
# fg = cv2.bitwise_and(og, og, mask=im_floodfill)
#
#
# # Display images.
# cv2.imshow("Thresholded Image", im_th)
# # cv2.imshow("Floodfilled Image", im_floodfill)
# # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
# # cv2.imshow("Foreground", fg)
# cv2.waitKey(0)


# img = cv2.imread('IMG_0073.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)
#
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
#
# rect = (50,50,2500,3500)
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
#
# black_mask = np.all(img == 0, axis=-1)
# alpha = np.uint8(np.logical_not(black_mask)) * 255
# result = np.dstack((img, alpha))  # Add the alpha channel
#
# cv2.imwrite('output.png', result)
# cv2.imshow("Thresholded Image", img)
# # # cv2.imshow("Floodfilled Image", im_floodfill)
# # # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
# # # cv2.imshow("Foreground", fg)
# cv2.waitKey(0)


# from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

def thresh_callback(val):
    threshold = val

    ## [Canny]
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    kernel = np.ones([30, 30])
    canny_output = cv.dilate(canny_output, kernel)
    # cv.floodFill(canny_output, None, (0, 0), 0)

    # canny_output = src_gray
    # cv.fastNlMeansDenoisingColored(canny_output, None, 100, 100, 7, 21)
    # cv2.floodFill(canny_output, None, seedPoint=(0, 0), newVal=(255, 255, 255), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
    # cv.fastNlMeansDenoisingColored(canny_output, None, 10, 10, 7, 21)
    # canny_output = cv.Canny(canny_output, threshold, threshold * 2)

    # canny_output = cv.cvtColor(canny_output, cv.COLOR_BGR2GRAY)
    # canny_output = cv.Canny(canny_output, threshold, threshold * 2)

    # [Canny]

    ## [findContours]
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ## [findContours]

    ## [allthework]
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    ## [allthework]

    ## [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]

    ## [forContour]
    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv.imshow('Contours?', canny_output)
        # cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    ## [forContour]

    ## [showDrawings]
    # Show in a window
    cv.imshow('Contours?', canny_output)
    cv.imshow('Contours', drawing)
    ## [showDrawings]

## [setup]
# Load source image
# parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
# args = parser.parse_args()

src = cv.imread("test_images/IMG_0173.jpg")
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)

# Convert image to gray and blur it
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_gray = cv.blur(src, (10,10))
src_gray = cv.blur(src, (2,2))
# src_gray = src
## [setup]

## [createWindow]
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src_gray)
## [createWindow]
## [trackbar]
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
## [trackbar]

cv.waitKey()
