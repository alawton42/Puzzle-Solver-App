import cv2 as cv


class PuzzlePiece:

    def __init__(self, piece_image):
        """
        Puzzle piece constructor.
        :param piece_image: np.array of a single puzzle piece, cropped and transparent background
        """
        self.original_piece_image = piece_image
        self.piece_orientations = self.__get_rotations()
        self.current_piece_index = 0

    def __get_rotations(self):
        """
        Generates each orientation of the original puzzle piece image.
        :return: array of images. Contains each 90 degree orientation of original_piece_image
        """
        rotations = [self.original_piece_image]
        for angle in [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]:
            rotations.append(cv.rotate(self.original_piece_image, angle))
        return rotations

    def get_edge_color(self):
        """
        Generate a 2d array which describes the colors on the edge of puzzle piece.
        [[north edge][east edge][south edge][west edge]]
        :return: 2d array containing the average color for edge of the puzzle piece
        """
        # Not necessary for assistant application
        pass

    def rotate(self):
        self.current_piece_index += 1
        self.current_piece_index %= 4
