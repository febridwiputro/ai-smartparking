import cv2
import numpy as np

def resize_with_aspect_ratio(image, target_size):
    """
    Resize an image while keeping the aspect ratio.

    Parameters:
    - image (np.ndarray): Input frame (image) in BGR format.
    - target_size (tuple): Maximum width and height (target_width, target_height).

    Returns:
    - resized_image (np.ndarray): Resized image with the same aspect ratio.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate the scaling factor to fit within the target size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a padded image with the target size and background color (black)
    padded_image = np.full((target_h, target_w, 3), (0, 0, 0), dtype=np.uint8)
    # Center the resized image on the padded background
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image

def create_grid(frames, rows, cols, frame_size, padding=5, bg_color=(0, 0, 0)):
    """
    Create a grid of frames (images) with aspect ratio preservation.

    Parameters:
    - frames (list): List of frames (images) in BGR format.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - frame_size (tuple): (width, height) for each cell in the grid.
    - padding (int): Space between frames in pixels.
    - bg_color (tuple): Background color for padding (BGR).

    Returns:
    - grid (np.ndarray): Final grid image.
    """
    if len(frames) != rows * cols:
        raise ValueError(f"Expected {rows * cols} frames, but got {len(frames)}.")

    # Resize each frame with aspect ratio preservation
    resized_frames = [resize_with_aspect_ratio(frame, frame_size) for frame in frames]

    # Get frame dimensions after resizing
    cell_width, cell_height = frame_size

    # Create an empty grid with padding and background color
    grid_height = rows * cell_height + (rows - 1) * padding
    grid_width = cols * cell_width + (cols - 1) * padding
    grid = np.full((grid_height, grid_width, 3), bg_color, dtype=np.uint8)

    # Populate the grid with frames
    for i in range(rows):
        for j in range(cols):
            y = i * (cell_height + padding)
            x = j * (cell_width + padding)
            grid[y:y + cell_height, x:x + cell_width] = resized_frames[i * cols + j]

    return grid

# Example usage:
if __name__ == "__main__":
    # Load sample frames (replace with your actual frames)
    frame1 = np.zeros((200, 300, 3), dtype=np.uint8)  # Black rectangle
    frame2 = np.ones((150, 400, 3), dtype=np.uint8) * 255  # White rectangle
    frame3 = np.ones((300, 150, 3), dtype=np.uint8) * 128  # Gray rectangle
    frame4 = np.ones((400, 250, 3), dtype=np.uint8) * 64  # Dark gray rectangle

    # Create a 2x2 grid with each frame fitting within (150x150) cells
    grid = create_grid([frame1, frame2, frame3, frame4], rows=2, cols=2, frame_size=(150, 150), padding=10)

    # Display the grid
    cv2.imshow("Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
