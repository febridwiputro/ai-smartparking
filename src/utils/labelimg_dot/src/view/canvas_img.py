import numpy as np
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


class CanvasImage(tk.Canvas):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._old_event = None
        self.config(bg="white")
        self.photo_image = None
        self.lasx = None
        self.lasy = None
        self.points = []
        self.label_teks = tk.StringVar()
        self.label_canvas_text = tk.StringVar()
        self.label_canvas_text_1 = tk.StringVar()
        self.image_container = None
        self.image = None
        self.image_original = None

        self.label = ttk.Label(self, textvariable=self.label_teks, border=5, relief=tk.SUNKEN)
        self.label.place(x=10, y=10)
        self.label_canvas = ttk.Label(self, textvariable=self.label_canvas_text, border=5, relief=tk.SUNKEN)
        self.label_canvas.place(x=10, y=30)

        self.label_canvas1 = ttk.Label(self, textvariable=self.label_canvas_text_1, border=5, relief=tk.SUNKEN)
        self.label_canvas1.place(x=10, y=50)
        self.mat_affine = np.eye(3)

        self.bind("<Button-3>", self.mouse_down_right)
        self.bind("<B3-Motion>", self.mouse_move_right)
        self.bind("<Motion>", self.mouse_move)
        self.bind("<Double-Button-3>", self.mouse_double_click_right)
        self.bind("<MouseWheel>", self.mouse_wheel)
        self.bind("<Enter>", lambda e: self.config(cursor="cross"))
        self.bind("<Leave>", lambda e: self.outside_canvas)
        self.bind("<Button-1>", lambda e: self.get_points(e))

    def show_img(self, path):
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.image_original = Image.fromarray(self.image)
        self.image = self.image_original.copy()
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.config(scrollregion=self.bbox(tk.ALL))
        self.label_teks.set(f"{self.image.width} x {self.image.height}")
        self.zoom_fit(self.image.width, self.image.height)
        self.draw_image(self.image)

    def outside_canvas(self):
        self.config(cursor="")
        self.label_canvas_text.set(f"(--,--)")
        self.label_canvas_text_1.set(f"(--,--)")

    def mouse_down_right(self, event):
        self._old_event = event

    def mouse_move_right(self, event):
        if self.image is None:
            return

        self.translate(event.x - self._old_event.x, event.y - self._old_event.y)
        self.redraw_image()
        self._old_event = event

    def mouse_move(self, event):
        if self.image is None:
            return

        self.label_canvas_text.set(f"({event.x} , {event.y})")
        image_point = self.to_image_point(event.x, event.y)

        self.label_canvas_text_1.set(
            f"({event.x / self.winfo_width()} , {event.y / self.winfo_height()})"
        )

        if len(image_point) > 0:
            self.label_teks.set(f"({image_point[0]:.2f}, {image_point[1]:.2f})")
        else:

            self.label_teks.set("(--, --)")

    def mouse_double_click_right(self, event):
        if self.image is None:
            return
        self.zoom_fit(self.image.width, self.image.height)
        self.redraw_image()

    def mouse_wheel(self, event):
        if event.state != 9:
            if event.delta < 0:
                self.scale_at(0.8, event.x, event.y)
            else:
                self.scale_at(1.25, event.x, event.y)
        self.redraw_image()

    def reset_transform(self):
        self.mat_affine = np.eye(3)

    def translate(self, offset_x, offset_y):
        mat = np.eye(3)
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)
        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale(self, scale: float):
        mat = np.eye(3)
        mat[0, 0] = scale
        mat[1, 1] = scale
        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale_at(self, scale: float, cx: float, cy: float):
        self.translate(-cx, -cy)
        self.scale(scale)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        if image_width * image_height <= 0 or canvas_width * canvas_height <= 0:
            return

        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        if canvas_width * image_height > image_width * canvas_height:
            scale = canvas_height / image_height
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            scale = canvas_width / image_width
            offsety = (canvas_height - image_height * scale) / 2

        self.scale(scale)
        self.translate(offsetx, offsety)

    def to_image_point(self, x, y):
        if self.image is None:
            return []

        mat_inv = np.linalg.inv(self.mat_affine)
        image_point = np.dot(mat_inv, (x, y, 1.))
        if (image_point[0] < 0 or image_point[1] < 0
                or image_point[0] > self.image.width
                or image_point[1] > self.image.height):

            return []
        return image_point

    def draw_image(self, pil_image):
        if pil_image is None:
            return

        self.image = pil_image
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        mat_inv = np.linalg.inv(self.mat_affine)
        affine_inv = (
            mat_inv[0, 0], mat_inv[0, 1], mat_inv[0, 2],
            mat_inv[1, 0], mat_inv[1, 1], mat_inv[1, 2]
        )

        dst = self.image.transform(
            (canvas_width, canvas_height),
            Image.AFFINE,
            affine_inv,
            Image.NEAREST
        )

        im = ImageTk.PhotoImage(image=dst)

        if self.image_container is None:
            self.image_container = self.create_image(0, 0, anchor='nw', image=im)
        else:
            self.itemconfig(self.image_container, image=im)

        self.photo_image = im

    def redraw_image(self):
        if self.image is None:
            return
        self.draw_image(self.image)

    def get_points(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", outline="black", tags="line_points")
        if len(self.points) > 1:
            self.create_line(self.points[-2][0], self.points[-2][1], x, y, fill="red", width=2)

        print(self.points)

