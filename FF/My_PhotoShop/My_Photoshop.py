import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ImageProcess import ImageProcessor

currentImage = []
panelA = None
panelB = None

root = Tk()
root.title('MyPhotoshoop')
root.geometry("900x400")


myMenu = Menu(root)
root.config(menu=myMenu)


def upload_file():
    global panelA, panelB

    global currentImage
    # f_types = [('Raw Files', '*.raw')]

    # filename = filedialog.askopenfilename(filetypes=f_types)
    path = filedialog.askopenfilename()
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        currentImage = image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        image = ImageTk.PhotoImage(image)

        if panelA is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
        else:
            panelA.configure(image=image)
            panelA.image = image
            if (panelB is not None):
                panelB.configure(image='')


def ShowImage(image):
    global panelB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image)

    image = ImageTk.PhotoImage(image)

    if panelB is None:
        panelB = Label(image=image)
        panelB.image = image
        panelB.pack(side="right", padx=10, pady=10)
    else:
        panelB.configure(image=image)
        panelB.image = image


def HasImage():
    global currentImage
    if (len(currentImage) > 0):
        return True
    else:
        messagebox.showerror(title='Hiba', message='Nincs feltöltve kép')
        return False


def Negate():
    global currentImage
    if (HasImage()):
        image = ImageProcessor.Negate(currentImage)
        ShowImage(image)


def GammaPrep():
    if (not HasImage()):
        return

    newWindow = Toplevel(root)
    newWindow.title("Gamma Érték")
    newWindow.geometry("300x75")

    Label(newWindow, text="Ird be a gamma értékét (0 - 10)",
          font=('Calibri 10')).pack()
    a = Entry(newWindow, width=35)
    a.pack()
    b = Button(master=newWindow, text="OK",
               command=lambda: click_p1(a.get(), newWindow))
    b.pack()


def click_p1(number, newWin):
    try:
        p1 = float(number)
        if (p1 > 0.0 and p1 <= 10.0):
            newWin.destroy()
            Gamma(p1)
        else:
            messagebox.showerror(
                title='Hiba', message='Nem megfelelő a tartomány 0 és 10 között')
    except:
        messagebox.showerror(title='Hiba', message='Nem szám az érték')


def Gamma(gamma):
    global currentImage
    if (gamma < 0 and gamma > 10):
        return

    if (HasImage()):
        image = ImageProcessor.create_gamma_lut(currentImage, gamma)
        ShowImage(image)


def Log():
    global currentImage
    if (HasImage()):
        image = ImageProcessor.Log(currentImage)
        ShowImage(image)


file_menu = Menu(myMenu)
myMenu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="New", command=lambda: upload_file())
file_menu.add_command(label="Negálás", command=lambda: Negate())
file_menu.add_command(label="Gamma Korelácio", command=lambda: GammaPrep())
file_menu.add_command(label="Logaritmus Transzformácio", command=lambda: Log())
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)


root.mainloop()
