import cv2 as cv
import numpy as np
import random as rng
from tkinter import Tk
from Puzzle_Piece import PuzzlePiece
# from Puzzle_App_GUI import GUI


def remove_background(img):
    """
    Remove flat color background of the image.
    References:
    https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
    :param img: original image
    :return: image with transparent background
    """
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Will need adjustment
    rect = (50, 50, 3000, 4000)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    black_mask = np.all(img == 0, axis=-1)
    alpha = np.uint8(np.logical_not(black_mask)) * 255
    result = np.dstack((img, alpha))  # Add the alpha channel

    return result


def separate_pieces(clean, num_pieces):
    """
    TBD
    :param clean:
    :return:
    """
    # Detect edges using Canny
    threshold = 100
    canny_output = cv.Canny(clean, threshold, threshold * 2)
    kernel = np.ones([20, 20])
    canny_output = cv.dilate(canny_output, kernel)
    # cv.floodFill(canny_output, None, (0, 0), 255)

    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons + get bounding rects
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Draw polygonal contour + bounding rects
    # Also store the area of each box
    area = []
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        # print((int(boundRect[i][0]), int(boundRect[i][1])),
        #       (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])))
        area.append((i, boundRect[i][2] * boundRect[i][3]))

    # Find largest boxes(likely the puzzle pieces) and crop them from the image
    area.sort(key=lambda x: x[1])
    cropped = []
    for _ in range(num_pieces):
        i, _ = area.pop()
        row1 = int(boundRect[i][0])
        row2 = int(boundRect[i][0] + boundRect[i][2])
        # print(row1, row2)
        col1 = int(boundRect[i][1])
        col2 = int(boundRect[i][1] + boundRect[i][3])
        # print(col1, col2)
        cropped_image = clean[col1:col2, row1:row2]
        cropped.append(cropped_image)
        # cv.imshow('Contours?', cropped_image)
        # cv.waitKey()
    return cropped


if __name__ == '__main__':
    # Demo for Intermediate Milestone #1
    # img = cv.imread('test_images/IMG_0173.jpg')
    # # img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # # cv.floodFill(img, None, seedPoint=(10, 10), newVal=(255, 255, 255), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
    # # cv.imshow("clean Image", img)
    # # cv.waitKey(0)
    # clean_image = remove_background(img)
    # cv.imwrite('test_images/removed_bg_test.png', clean_image)
    # cv.imshow("clean Image", clean_image)
    # cv.waitKey(0)

    # separate_pieces(clean_image)

    # img = cv.imread('test_images/Pieces_for_milestone2_demo/test_image_for_rotation.png')
    # test_piece = PuzzlePiece(img)
    # for i, r in enumerate(test_piece.piece_orientations):
    #     cv.imshow("tmp", r)
    #     cv.imwrite(f'test_images/output/rotated{i}.png', r)
    #     cv.waitKey(0)

    # Demo for Intermediate Milestone #2
    # import glob
    #
    # pieces = []
    # for filename in glob.glob('test_images/Pieces_for_milestone2_demo/*.png'):
    #     img = cv.imread(filename)
    #     # cv.imshow("tmp", img)
    #     # cv.waitKey(0)
    #     pieces.append(PuzzlePiece(img))
    #
    # root = Tk()
    # root.geometry("800x600")
    # root.configure(background='black')
    # draw = GUI(root)
    #
    # draw.run()

    img = cv.imread('Dataset/PNGImages/0010.png')
    img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv.floodFill(img, None, seedPoint=(10, 10), newVal=(255, 255, 255), loDiff=(3, 3, 3, 3), upDiff=(3, 3, 3, 3))
    clean_image = remove_background(img)
    # cv.imshow("clean Image", clean_image)
    # cv.waitKey()
    separate_pieces(clean_image, 6)

