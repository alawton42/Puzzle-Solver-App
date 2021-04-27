
import cv2 as cv
from PIL import Image, ImageTk
from Puzzle_Piece import PuzzlePiece
from Puzzle_Solver import remove_background, separate_pieces

try:
    from tkinter import Tk, Button, Menu, Frame, Label, simpledialog
    from tkinter import filedialog
    import pickle
    from functools import partial

except ImportError:
    print("tkinter did not import successfully - check you are running Python 3 and that tkinter is available.")
    exit(1)


class GUI:

    def __init__(self, root):
        self.root = root
        self.pieces = []

        self.file_name = None

        self.selected = False
        self.currently_selected = -1

        self.num_col = 10
        self.sensitivity = 3

        # Set up the root window
        self.root.title("Puzzle Assistant")

        # Define our layout grid
        self.grid = Frame(self.root, bg="#000000")
        self.grid.grid_columnconfigure(self.num_col, weight=1)
        self.grid.grid_rowconfigure(13, weight=1)
        self.title = ""

        # Top level menu buttons
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open", command=self.open)
        self.root.config(menu=menubar)

        # Initial display
        self.title = Label(self.grid, text="Puzzle Assistant", fg="white", bg="black", font=('Arial', 25), pady=30)
        self.title.grid(row=0, column=0, columnspan=self.num_col)
        start_up_text = Label(self.grid, text="Select an image using the file menu or button below.",
                              fg="white", bg="black", font=('Arial', 18), pady=20)
        start_up_text.grid(row=2, column=0, columnspan=self.num_col)
        start_up_button = Button(self.grid, text="Choose image", command=self.open)
        start_up_button.grid(row=3, column=0, columnspan=self.num_col)

        col_button = Button(self.grid, text="Change Layout size", command=self.change_layout_size)
        col_button.grid(row=12, column=0, columnspan=self.num_col, pady=5, padx=5)

        sensitivity_button = Button(self.grid, text="Change image sensitivity", command=self.change_sensitivity)
        sensitivity_button.grid(row=13, column=0, columnspan=self.num_col, pady=5, padx=5)

        self.grid.pack()

        # prep for image to be imported
        self.pieces_buttons = []
        self.rotate_buttons = []
        self.piece_image = []

    def move(self, i):
        if not self.selected:
            self.selected = True
            self.currently_selected = i
        else:
            # Get all info for the swap
            tmp1 = self.rotate_buttons[self.currently_selected]
            tmp2 = self.pieces_buttons[self.currently_selected]
            tmp3 = self.rotate_buttons[i]
            tmp4 = self.pieces_buttons[i]
            tmp5 = self.pieces[i]
            tmp6 = self.pieces[self.currently_selected]

            # Swap
            self.rotate_buttons[i] = tmp1
            self.pieces_buttons[i] = tmp2
            self.pieces[i] = tmp6
            self.rotate_buttons[self.currently_selected] = tmp3
            self.pieces_buttons[self.currently_selected] = tmp4
            self.pieces[self.currently_selected] = tmp5

            # Reset swap selection
            self.selected = False
            self.currently_selected = -1

            # redo layout with the swap
            self.layout()

    def rotate(self, i):
        self.pieces[i].rotate()
        self.layout()

    def layout(self):
        # Clear layout
        self.grid.destroy()

        # Remake Title and layout structure
        self.grid = Frame(self.root, bg="#000000")
        self.grid.grid_columnconfigure( self.num_col, weight=1)
        self.grid.grid_rowconfigure( self.num_col, weight=1)
        self.title = Label(self.grid, text="Puzzle Assistant", fg="white", bg="black", font=('Arial', 25), pady=30)
        self.title.grid(row=0, column=0, columnspan= self.num_col)

        self.piece_image = []

        for i, piece in enumerate(self.pieces):
            # Display piece
            img = cv.resize(piece.piece_orientations[piece.current_piece_index], (100, 100))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            self.piece_image.append(ImageTk.PhotoImage(Image.fromarray(img)))
            tmp = Button(self.grid, image=self.piece_image[i], command=partial(self.move, i))
            tmp.grid(row=(i//self.num_col) + (i//self.num_col) + 3, column=i % self.num_col)
            self.pieces_buttons.append(tmp)

            # Rotate buttons
            tmp2 = Button(self.grid, text="rotate", command=partial(self.rotate, i))
            tmp2.grid(row=(i//self.num_col+1 + (i//self.num_col)) + 3, column=i % self.num_col)
            self.rotate_buttons.append(tmp2)

        col_button = Button(self.grid, text="Layout size", command=self.change_layout_size)
        col_button.grid(row=13, column=0, pady=5, padx=5)

        sensitivity_button = Button(self.grid, text="Image sensitivity", command=self.change_sensitivity)
        sensitivity_button.grid(row=13, column=1, pady=5, padx=5)

        self.grid.pack()

    def open(self):
        self.file_name = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])

        img = cv.imread(self.file_name)

        # Pre-processing image
        img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv.floodFill(img, None, seedPoint=(10, 10), newVal=(255, 255, 255),
                     loDiff=(self.sensitivity, self.sensitivity, self.sensitivity, self.sensitivity),
                     upDiff=(self.sensitivity, self.sensitivity, self.sensitivity, self.sensitivity))
        # remove background
        clean_image = remove_background(img)
        # Prompt for number of pieces
        num_pieces = simpledialog.askinteger("Input", "How many pieces are in the selected image?",
                                             parent=self.root, minvalue=0, maxvalue=100)
        # Separate pieces
        piece_images = separate_pieces(clean_image, num_pieces)

        # Store pieces for layout
        self.pieces = []
        for image in piece_images:
            tmp = PuzzlePiece(image)
            self.pieces.append(tmp)

        self.layout()

    def change_layout_size(self):
        # Prompt for number of columns
        self.num_col = simpledialog.askinteger("Input", "Enter the number of columns (maximum:13)",
                                               parent=self.root, minvalue=2, maxvalue=13)
        self.layout()

    def change_sensitivity(self):
        self.sensitivity = simpledialog.askinteger("Input", "Enter Background removal sensitivity(2-6):",
                                                   parent=self.root, minvalue=2, maxvalue=6)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    root = Tk()
    root.geometry("1400x1000")
    root.configure(background='black')
    draw = GUI(root)

    draw.run()
