import tkinter as tk
from tkinter import messagebox

def alert_popup(title, message):
    root = tk.Tk()
    root.withdraw()  # Hide the small tk window
    messagebox.showwarning(title, message)
    root.destroy()





