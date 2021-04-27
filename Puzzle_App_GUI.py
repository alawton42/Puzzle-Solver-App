
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
        self.unsaved = False

        self.selected = False
        self.currently_selected = -1

        # Set up the root window
        self.root.title("Puzzle Assistant")

        # Define our layout grid
        self.grid = Frame(self.root, bg="#000000")
        self.grid.grid_columnconfigure(10, weight=1)
        self.grid.grid_rowconfigure(10, weight=1)
        self.title = ""

        # Top level menu buttons
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open", command=self.open)
        self.root.config(menu=menubar)

        self.title = Label(self.grid, text="Puzzle Assistant", fg="white", bg="black", font=('Arial', 25), pady=30)
        self.title.grid(row=0, column=0, columnspan=10)

        start_up_text = Label(self.grid, text="Select an image using the file menu or button below.",
                              fg="white", bg="black", font=('Arial', 18), pady=20)
        start_up_text.grid(row=2, column=0, columnspan=10)

        start_up_button = Button(self.grid, text="Choose image", command=self.open)
        start_up_button.grid(row=3, column=0, columnspan=10)

        self.grid.pack()

        self.pieces_buttons = []
        self.rotate_buttons = []
        self.piece_image = []

        # self.layout()

    def move(self, i):
        if not self.selected:
            self.selected = True
            self.currently_selected = i
        else:
            tmp1 = self.rotate_buttons[self.currently_selected]
            # tmp1.grid(row=2, column=i)
            # tmp1.configure(command=partial(self.rotate, i))

            tmp2 = self.pieces_buttons[self.currently_selected]
            # tmp2.grid(row=1, column=i)
            # tmp2.configure(command=partial(self.move, i))

            tmp3 = self.rotate_buttons[i]
            # tmp3.grid(row=2, column=self.currently_selected)
            # tmp3.configure(command=partial(self.rotate, self.currently_selected))

            tmp4 = self.pieces_buttons[i]
            # tmp4.grid(row=1, column=self.currently_selected)
            # tmp4.configure(command=partial(self.move, self.currently_selected))

            tmp5 = self.pieces[i]
            tmp6 = self.pieces[self.currently_selected]

            self.rotate_buttons[i] = tmp1
            self.pieces_buttons[i] = tmp2
            self.pieces[i] = tmp6
            self.rotate_buttons[self.currently_selected] = tmp3
            self.pieces_buttons[self.currently_selected] = tmp4
            self.pieces[self.currently_selected] = tmp5

            self.selected = False
            self.currently_selected = -1

            self.layout()

    def rotate(self, i):
        self.pieces[i].rotate()
        img = cv.resize(self.pieces[i].piece_orientations[self.pieces[i].current_piece_index], (100, 100))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # self.piece_image[i] = ImageTk.PhotoImage(Image.fromarray(img))
        self.layout()
        # for j, p in enumerate(self.piece_image):
        #     self.pieces_buttons[j].configure(image=p)

    def layout(self):
        self.grid.destroy()

        self.grid = Frame(self.root, bg="#000000")
        self.grid.grid_columnconfigure(10, weight=1)
        self.grid.grid_rowconfigure(10, weight=1)

        self.title = Label(self.grid, text="Puzzle Assistant", fg="white", bg="black", font=('Arial', 25), pady=30)
        self.title.grid(row=0, column=0, columnspan=10)

        self.piece_image = []

        for i, piece in enumerate(self.pieces):
            img = cv.resize(piece.piece_orientations[piece.current_piece_index], (100, 100))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            self.piece_image.append(ImageTk.PhotoImage(Image.fromarray(img)))
            tmp = Button(self.grid, image=self.piece_image[i], command=partial(self.move, i))
            tmp.grid(row=(i//10) + (i//10) + 3, column=i % 10)
            self.pieces_buttons.append(tmp)

            tmp2 = Button(self.grid, text="rotate", command=partial(self.rotate, i))
            tmp2.grid(row=(i//10+1 + (i//10)) + 3, column=i % 10)
            self.rotate_buttons.append(tmp2)

        self.grid.pack()

    def open(self):
        self.file_name = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])

        self.grid.destroy()

        self.grid = Frame(self.root, bg="#000000")
        self.grid.grid_columnconfigure(10, weight=1)
        self.grid.grid_rowconfigure(10, weight=1)

        self.title = Label(self.grid, text="Puzzle Assistant", fg="white", bg="black", font=('Arial', 25), pady=30)
        self.title.grid(row=0, column=0, columnspan=7)

        start_up_text = Label(self.grid, text="Loading image...",
                              fg="white", bg="black", font=('Arial', 18), pady=20)
        start_up_text.grid(row=2, column=0, columnspan=10)
        self.grid.pack()

        img = cv.imread(self.file_name)

        start_up_text.configure(text="Processing Image...")
        self.grid.pack()

        img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv.floodFill(img, None, seedPoint=(10, 10), newVal=(255, 255, 255), loDiff=(3, 3, 3, 3), upDiff=(3, 3, 3, 3))
        clean_image = remove_background(img)
        num_pieces = simpledialog.askinteger("Input", "How many pieces are in the selected image?",
                                         parent=self.root,
                                         minvalue=0, maxvalue=100)
        piece_images = separate_pieces(clean_image, num_pieces)
        self.pieces = []
        for image in piece_images:
            tmp = PuzzlePiece(image)
            self.pieces.append(tmp)

        self.layout()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    root = Tk()
    root.geometry("1200x1000")
    root.configure(background='black')
    draw = GUI(root)

    draw.run()
