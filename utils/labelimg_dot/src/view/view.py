import tkinter as tk
from tkinter import filedialog
import os
import cv2
from PIL import Image, ImageTk
import tkinter.ttk as ttk


class Labeling:
    def __init__(self, _root: tk.Tk):
        self.event_button_list = []
        self.points = []
        self.photo_image = None
        self.root = _root
        self.image_container = None
        self.image = None
        self.image_original = None
        self.last_x = None
        self.last_y = None

        def on_canvas(canvas: tk.Canvas):
            canvas.config(cursor="cross")

        def outside_canvas(canvas: tk.Canvas):
            canvas.config(cursor="")

        self.root.config(bg="skyblue")
        self.root.title("Labeling")
        self.root.geometry("1080x720")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.bind("<q>", lambda e: self.root.quit())
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")))
        self.opened = False
        self.path = ""

        # Frame for button
        self.frame_button = ttk.Frame(self.root, width=100, height=50)

        self.frame_button.pack(expand=True, fill="both")

        self.teks_path = ttk.Label(self.root, text=self.path)
        self.teks_path.pack()


        # Canvas for displaying image from camera
        self.canvas = tk.Canvas(self.frame_button, width=640, height=480)
        self.canvas.pack(expand=True, fill="both")
        self.canvas.place(x=10, y=50)
        self.canvas.config(bg="white")
        self.canvas.bind("<MouseWheel>", self.zoom_image)
        self.canvas.bind("<Button-1>", self.get_points)
        self.canvas.bind("<Button-3>", self.save_points)
        self.canvas.bind("<Enter>", lambda e: on_canvas(self.canvas))
        self.canvas.bind("<Leave>", lambda e: outside_canvas(self.canvas))

    def show_img(self):
        image = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        self.image_original = Image.fromarray(image)
        self.image = self.image_original.copy()
        self.photo_image = ImageTk.PhotoImage(self.image)

        if self.image_container is None:
            self.image_container = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        else:
            self.canvas.itemconfig(self.image_container, image=self.photo_image)

    def get_points(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", outline="green", tags="line_points")
        if len(self.points) > 1:
            self.canvas.create_line(self.points[-2][0], self.points[-2][1], x, y, fill="red", width=2)

        print(self.points)

    def save_points(self, event):
        self.event_button_list.append(self.points.copy())
        self.points = []
        self.canvas.delete('line_points')

    def view_points(self):
        if self.path != "":
            self.show_img()
        if len(self.event_button_list) > 0:
            for points in self.event_button_list:
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        self.canvas.create_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                                                fill="red", width=2)

    def zoom_image(self, event):
        if self.image is not None:
            scale_factor = 1.1 if event.delta > 0 else 0.9
            new_size = (int(self.image.width * scale_factor), int(self.image.height * scale_factor))
            self.image = self.image_original.resize(new_size, Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(self.image)
            self.canvas.itemconfig(self.image_container, image=self.photo_image)
            self.canvas.config(scrollregion=self.canvas.bbox(self.image_container))
            print(self.image.width, self.image.height)



    def reset_last_position(self, event):
        self.last_x, self.last_y = None, None

    def loop(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    Labeling(_root=root).loop()
