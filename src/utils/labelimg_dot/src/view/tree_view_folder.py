import tkinter.ttk as ttk
import tkinter as tk
from tkinter import filedialog
import os


class TreeViewModel(ttk.Treeview):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.directory_tree = None
        self.path_image = None
        self.pack()
        self.heading("#0", text="Open Folder")
        self.column("#0", width=300)
        self.folder_img = tk.PhotoImage(file=r"C:\Users\eril.sanjaya\Documents\smart-parking\utils\labelimg_dot\resource\folder_icon.png")
        self.photo_img = tk.PhotoImage(file=r"C:\Users\eril.sanjaya\Documents\smart-parking\utils\labelimg_dot\resource\image_icon.png")
        self.bind("<Double-1>", self.on_double_click)

    def on_double_click(self, event):
        region = self.identify("region", event.x, event.y)
        if region == "heading":
            self.open_folder()

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.delete(*self.get_children())
            self.directory_tree = DirectoryTree(folder_path)
            try:
                self.directory_tree.generate_tree()
                self.directory_tree.display_tree()

                self.generate_tree(folder_path, '')
            except PermissionError as pe:
                print(pe)

    def generate_tree(self, dir_path, parent_node):

        for item in os.listdir(dir_path):
            item_path = os.path.normpath(os.path.join(dir_path, item))
            if os.path.isdir(item_path):
                node = self.insert(parent_node, 'end', text=item, open=False, image=self.folder_img)
                self.generate_tree(item_path, node)
            else:
                if item.split('.')[-1] in ['jpg', 'png', 'jpeg']:
                    self.insert(parent_node, 'end', text=item, image=self.photo_img)


class DirectoryTree:
    def __init__(self, root_path):
        self.root_path = root_path
        self.tree = {}

    def generate_tree(self):
        self.tree = self._generate_tree(self.root_path)

    def _generate_tree(self, path):
        tree = {}
        for item in os.listdir(path):

            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                tree[item] = self._generate_tree(item_path)
            else:
                tree[item] = None
        return tree

    def display_tree(self):
        self._display_tree(self.tree)

    def _display_tree(self, tree, level=0):
        for item, subtree in tree.items():
            if subtree:
                self._display_tree(subtree, level + 1)
