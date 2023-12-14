from tkinter import *
from tkinter import messagebox

w = Tk()

w.geometry('200x250')


def buttons():
    r1 = Toplevel(w)
    r1.geometry('200x250')
    labelexample = Label(r1, text='GOOD')
    labelexample.pack()
    b = Button(master=r1, text="SIGN UP", command=buttons)
    b.pack()


t = Label(text="MIB", font=("Arial", 49))
t.pack(side=TOP)

e = Label(text="Email")
e1 = Entry()

e.pack()
e1.pack()

p = Label(text="Password")
p.pack()

p1 = Entry()
p1.pack()

b = Button(text="SIGN UP", command=buttons)
b.pack()
w.mainloop()
