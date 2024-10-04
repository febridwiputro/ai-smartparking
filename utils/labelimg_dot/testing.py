import tkinter as tk

root = tk.Tk()
root.geometry("800x600")

paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

sidebar = tk.Frame(paned_window, bg="lightgrey", width=200)
sidebar.pack(fill=tk.Y, side=tk.LEFT, expand=False)

main_area = tk.Frame(paned_window, bg="white")
main_area.pack(fill=tk.BOTH, expand=True)

paned_window.add(sidebar)
paned_window.add(main_area)

root.mainloop()
