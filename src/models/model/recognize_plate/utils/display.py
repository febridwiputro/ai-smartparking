import logging, os
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

def display_character_segments(char, char_resized, is_show=False):
    if is_show:
        fig = plt.figure(figsize=(10, 4.5))
        grid = gridspec.GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.imshow(char, cmap="gray")
        ax1.set_title("Original Character Segment", loc='left')
        ax1.axis(False)

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.imshow(char_resized, cmap="gray")
        ax2.set_title("Resized Character", loc='left')
        ax2.axis(False)

        plt.tight_layout()
        plt.show()

def display_results(rgb, inv, segmented_image, crop_characters, final_string, result_string, is_save=False, is_show=False):
    fig = plt.figure(figsize=(10, 7.5))
    grid = gridspec.GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    ax1.set_title("Original Image", loc='left')
    ax1.axis(False)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(inv)
    ax2.set_title("Inverse Image", loc='left')
    ax2.axis(False)

    ax3 = fig.add_subplot(grid[1, 0])
    ax3.imshow(segmented_image)
    ax3.set_title(f"IDENTIFIED CHARACTER: {len(crop_characters)}", loc='left')
    ax3.axis(False)

    if len(crop_characters) > 0:
        char_grid = gridspec.GridSpecFromSubplotSpec(1, len(crop_characters), subplot_spec=grid[1, 1])
        for i in range(len(crop_characters)):
            ax = fig.add_subplot(char_grid[i])
            ax.imshow(crop_characters[i], cmap="gray")
            ax.axis(False)
            if i == 0:
                ax.set_title("Segmented Characters", loc='left')
    else:
        ax4 = fig.add_subplot(grid[2, 0])
        ax4.set_title("No characters found", loc='left', fontweight="bold")
        ax4.axis(False)

    ax5 = fig.add_subplot(grid[2, 1])
    ax5.set_title(f"Plate No: {final_string}", loc='left', fontweight="bold", fontsize=20)
    ax5.axis(False)
    ax5.imshow(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    ax6 = fig.add_subplot(grid[2, 0])
    ax6.set_title("Character Detection Results", loc='left', fontweight="bold")
    ax6.text(0, 1, result_string, ha='left', va='top', fontsize=10)
    ax6.axis(False)

    plt.tight_layout()

    # Save the figure if is_save is True
    if is_save:
        # Create the directory if it does not exist
        directory = "grid_character_recognition"
        os.makedirs(directory, exist_ok=True)
        
        # Create the filename with a timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        filename = f"{directory}/{timestamp}.jpg"
        
        plt.savefig(filename)  # Save the image
        # print(f"Results image saved as '{filename}'.")

    # Show the figure if is_show is True
    if is_show:
        plt.show()

    # If neither is set, just close the plot to avoid displaying it
    if not is_save and not is_show:
        plt.close(fig)