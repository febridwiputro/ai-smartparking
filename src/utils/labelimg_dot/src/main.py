from view.tree_view_folder import TreeViewModel, os
from view.canvas_img import CanvasImage
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import cv2
from PIL import Image, ImageTk


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Label img")
        self.state('zoomed')
        self.minsize(int(self.winfo_screenwidth() / 2), int(self.winfo_screenheight() / 2))



        self.panned_window = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.panned_window.pack(expand=True, fill="both")

        self.frame_canvas = ttk.Frame(self,  border=5, relief=tk.SUNKEN)
        self.frame_canvas.pack(expand=True, fill="both", padx=5, pady=5, side=tk.RIGHT)

        self.frame_tree_view = ttk.Frame(self,  border=5, relief=tk.SUNKEN)
        self.frame_tree_view.pack(expand=True, fill="both", padx=5, pady=5, side=tk.LEFT)

        self.panned_window.add(self.frame_tree_view)
        self.panned_window.add(self.frame_canvas)

        self.canvas = CanvasImage(self.frame_canvas)
        self.canvas.pack(expand=True, fill="both", padx=30, pady=30)

        self.tree_view = TreeViewModel(self.frame_tree_view)
        self.tree_view.pack(expand=True, fill="both", pady=(5, 20), padx=5)

        self.frame_sidebar = ttk.Frame(self, width=int(self.winfo_screenwidth() * 0.03), border=5, relief=tk.SUNKEN)
        self.frame_sidebar.pack(expand=True, fill="both", padx=5, pady=5, side=tk.LEFT, anchor=tk.W)

        self.menu = tk.Menu(self)
        self.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Folder", command=self.tree_view.open_folder)

        self.tree_view.bind("<<TreeviewSelect>>", self.on_open_file)

    def on_open_file(self, e):
        item = self.tree_view.selection()[0] if self.tree_view.selection() else None
        parent_iid = self.tree_view.parent(item)
        node = []
        while parent_iid != '':
            node.insert(0, self.tree_view.item(parent_iid)['text'])
            parent_iid = self.tree_view.parent(parent_iid)
        i = self.tree_view.item(item, "text")

        path = os.path.join(*node, i)
        if path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            self.canvas.show_img(path)




if __name__ == "__main__":
    root = MainApp()
    root.mainloop()


"""
1. Titik harus dalam gambar bahkan habis di zoom/pan
2. line harus ikut berubah ketika gambar zoom/pan
3. koordinat akan tetap sama bahkan jika gambar zoom/pan
4. line mengikuti gambar
5. titik mengikuti gambar
6. titik dan line harus bisa dihapus

"""