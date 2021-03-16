import cv2 as cv
import numpy as np
from Puzzle_Piece import PuzzlePiece


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
    rect = (50, 50, 2500, 3500)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    black_mask = np.all(img == 0, axis=-1)
    alpha = np.uint8(np.logical_not(black_mask)) * 255
    result = np.dstack((img, alpha))  # Add the alpha channel

    return result

def seperate_pieces(img):
    """
    TBD
    :param img:
    :return:
    """
    pass


if __name__ == '__main__':
    # Demo for Intermediate Milestone #1
    # img = cv.imread('test_images/IMG_0073.jpg')
    # clean_image = remove_background(img)
    # cv.imwrite('test_images/removed_bg_test.png', clean_image)
    # cv.imshow("clean Image", clean_image)
    # cv.waitKey(0)

    img = cv.imread('test_images/test_image_for_rotation.png')
    test_piece = PuzzlePiece(img)
    for i, r in enumerate(test_piece.piece_orientations):
        cv.imshow("tmp", r)
        cv.imwrite(f'test_images/output/rotated{i}.png', r)
        cv.waitKey(0)


