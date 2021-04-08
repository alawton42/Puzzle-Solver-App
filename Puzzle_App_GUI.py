
import cv2 as cv
from PIL import Image, ImageTk

try:
    from tkinter import Tk, Button, Menu, Frame, Label
    from tkinter import filedialog
    import pickle
    from functools import partial

except ImportError:
    print("tkinter did not import successfully - check you are running Python 3 and that tkinter is available.")
    exit(1)


class GUI:

    def __init__(self, root, pieces):
        self.root = root
        self.pieces = pieces

        self.file_name = None
        self.unsaved = False

        self.selected = False
        self.currently_selected = -1

        # Set up the root window
        self.root.title("Puzzle Assistant")

        # Define our layout grid
        self.grid = Frame(self.root, bg="#000000")
        self.grid.grid_columnconfigure(7, weight=1)
        self.grid.grid_rowconfigure(3, weight=1)

        # Top level menu buttons
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        # menubar.add_command(label="File")
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        self.title = Label(self.grid, text="Puzzle Assistant", fg="white", bg="black", font=('Arial', 25), pady=30)
        self.title.grid(row=0, column=0, columnspan=7)

        self.pieces_buttons = []
        self.rotate_buttons = []
        self.piece_image = []

        for i, piece in enumerate(self.pieces):
            img = cv.resize(piece.piece_orientations[piece.current_piece_index], (100, 100))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            self.piece_image.append(ImageTk.PhotoImage(Image.fromarray(img)))
            tmp = Button(self.grid, image=self.piece_image[i], command=partial(self.move, i))
            tmp.grid(row=1, column=i)
            self.pieces_buttons.append(tmp)

            tmp2 = Button(self.grid, text="rotate", command=partial(self.rotate, i))
            tmp2.grid(row=2, column=i)
            self.rotate_buttons.append(tmp2)

        self.grid.pack()

    def move(self, i):
        if not self.selected:
            self.selected = True
            self.currently_selected = i
        else:
            tmp1 = self.rotate_buttons[self.currently_selected]
            tmp1.grid(row=2, column=i)
            tmp1.configure(command=partial(self.rotate, i))

            tmp2 = self.pieces_buttons[self.currently_selected]
            tmp2.grid(row=1, column=i)
            tmp2.configure(command=partial(self.move, i))

            tmp3 = self.rotate_buttons[i]
            tmp3.grid(row=2, column=self.currently_selected)
            tmp3.configure(command=partial(self.rotate, self.currently_selected))

            tmp4 = self.pieces_buttons[i]
            tmp4.grid(row=1, column=self.currently_selected)
            tmp4.configure(command=partial(self.move, self.currently_selected))

            self.rotate_buttons[i] = tmp1
            self.pieces_buttons[i] = tmp2
            self.rotate_buttons[self.currently_selected] = tmp3
            self.pieces_buttons[self.currently_selected] = tmp4

            self.selected = False
            self.currently_selected = -1

    def rotate(self, i):
        self.pieces[i].rotate()
        img = cv.resize(self.pieces[i].piece_orientations[self.pieces[i].current_piece_index], (100, 100))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.piece_image[i] = ImageTk.PhotoImage(Image.fromarray(img))
        self.pieces_buttons[i].configure(image=self.piece_image[i])

    def run(self):
        self.root.mainloop()


