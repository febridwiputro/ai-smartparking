import os
import cv2
from tqdm import tqdm

class ImageEditor:
    def __init__(self, target_size_kb=150, step=5, min_quality=10, output_folder='compressed_images'):
        """
        Initialize the ImageEditor with default parameters for compression and cropping.

        Parameters:
        - target_size_kb: int, target size in KB for each image (default 150 KB).
        - step: int, decrement step for quality adjustment (default 5).
        - min_quality: int, minimum quality to avoid over-compression (default 10).
        - output_folder: str, directory to save processed images (default 'compressed_images').
        """
        self.target_size_kb = target_size_kb
        self.step = step
        self.min_quality = min_quality
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def compress_image(self, input_path, output_path):
        """Compress a single image to the target size by adjusting JPEG quality."""
        img = cv2.imread(input_path)
        quality = 95  # Start with high quality

        while quality >= self.min_quality:
            # Compress image and save it
            cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            size_kb = os.path.getsize(output_path) / 1024  # Get size in KB

            if size_kb <= self.target_size_kb:
                print(f"Compressed {os.path.basename(input_path)} to {size_kb:.2f} KB with quality {quality}.")
                break

            quality -= self.step  # Decrease quality and try again

        if quality < self.min_quality:
            print(f"Warning: Could not compress {os.path.basename(input_path)} to {self.target_size_kb} KB.")

    def crop_image(self, input_path, output_path, size=(640, 640)):
        """Crop a single image to the specified size (default 640x640)."""
        img = cv2.imread(input_path)
        img_cropped = self._center_crop(img, size)
        cv2.imwrite(output_path, img_cropped)
        print(f"Cropped {os.path.basename(input_path)} to {size}.")

    def crop_images_in_folder(self, folder_path, output_folder=None, size=(640, 640)):
        """Crop all images in a folder to the specified size, saving to a new folder."""
        if output_folder is None:
            output_folder = os.path.join(folder_path, 'cropped_images')
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in tqdm(os.listdir(folder_path), desc="Cropping Images"):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(folder_path, file_name)
                output_path = os.path.join(output_folder, file_name)
                self.crop_image(input_path, output_path, size)

    def _center_crop(self, img, size):
        """Center-crop the image to the specified size."""
        height, width = img.shape[:2]
        target_width, target_height = size

        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return img[top:bottom, left:right]

    def compress_images_in_folder(self, folder_path):
        """Compress all images in a specified folder to the target size."""
        for file_name in tqdm(os.listdir(folder_path), desc="Compressing Images"):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(folder_path, file_name)
                output_path = os.path.join(self.output_folder, file_name)
                self.compress_image(input_path, output_path)

    def adjust_image_size(self, input_path, output_path, target_size=(640, 640)):
        """
        Adjust the image size to the target size by adding a black background.

        Parameters:
        - input_path: str, path to the input image.
        - output_path: str, path to save the adjusted image.
        - target_size: tuple, target size in pixels (default (640, 640)).
        """
        img = cv2.imread(input_path)
        height, width = img.shape[:2]
        target_width, target_height = target_size

        # Create a new image with a black background
        new_img = cv2.cvtColor(cv2.resize(img, (target_width, target_height)), cv2.COLOR_BGR2RGB)

        # Calculate position to paste the original image
        left = (target_width - width) // 2
        top = (target_height - height) // 2
        
        # Create a black image and place the original image in the center
        adjusted_img = cv2.copyMakeBorder(img, top, target_height - height - top, left, target_width - width - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cv2.imwrite(output_path, adjusted_img)
        print(f"Adjusted {os.path.basename(input_path)} to {target_size}.")

    def adjust_images_in_folder(self, folder_path, output_folder=None, target_size=(640, 640)):
        """Adjust all images in a folder to the specified size by adding a black background."""
        if output_folder is None:
            output_folder = os.path.join(folder_path, 'adjusted_images')
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in tqdm(os.listdir(folder_path), desc="Adjusting Images"):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(folder_path, file_name)
                output_path = os.path.join(output_folder, file_name)
                self.adjust_image_size(input_path, output_path, target_size)


# Example usage
# folder_path = r"D:\plate_upload\G"
# folder_path = r"C:\Users\DOT\Documents\febri\github\ai-smartparking\compressed_images"
folder_path = r"D:\plat_unregister"
editor = ImageEditor(target_size_kb=120)  # Set target size to 100 KB

# # Adjust images in a folder to 640x640, saving them in a new folder
# editor.adjust_images_in_folder(folder_path, output_folder=os.path.join(folder_path, 'adjusted_images'), target_size=(640, 640))

# # Crop images in a folder to 640x640, saving them in a new folder
# editor.crop_images_in_folder(folder_path, output_folder=os.path.join(folder_path, 'cropped_images'), size=(640, 640))

# Compress images in a folder
editor.compress_images_in_folder(folder_path)



# import argparse
# from PIL import Image
# import os

# class ImageEditor:
#     def __init__(self, target_size_kb=150, step=5, min_quality=10, output_folder='processed_images'):
#         """
#         Initialize the ImageEditor with default parameters for compression and cropping.

#         Parameters:
#         - target_size_kb: int, target size in KB for each image (default 150 KB).
#         - step: int, decrement step for quality adjustment (default 5).
#         - min_quality: int, minimum quality to avoid over-compression (default 10).
#         - output_folder: str, directory to save processed images (default 'processed_images').
#         """
#         self.target_size_kb = target_size_kb
#         self.step = step
#         self.min_quality = min_quality
#         self.output_folder = output_folder

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#     def compress_image(self, input_path, output_path):
#         """Compress a single image to the target size by adjusting JPEG quality."""
#         img = Image.open(input_path)
#         quality = 95  # Start with high quality

#         while quality >= self.min_quality:
#             img.save(output_path, 'JPEG', quality=quality)
#             size_kb = os.path.getsize(output_path) / 1024  # Get size in KB

#             if size_kb <= self.target_size_kb:
#                 print(f"Compressed {os.path.basename(input_path)} to {size_kb:.2f} KB with quality {quality}.")
#                 break

#             quality -= self.step  # Decrease quality and try again

#         if quality < self.min_quality:
#             print(f"Warning: Could not compress {os.path.basename(input_path)} to {self.target_size_kb} KB.")

#     def crop_image(self, input_path, output_path, size=(640, 640)):
#         """Crop a single image to the specified size (default 640x640)."""
#         img = Image.open(input_path)
#         img_cropped = self._center_crop(img, size)
#         img_cropped.save(output_path)
#         print(f"Cropped {os.path.basename(input_path)} to {size}.")

#     def crop_images_in_folder(self, folder_path, size=(640, 640)):
#         """Crop all images in a folder to the specified size."""
#         for file_name in os.listdir(folder_path):
#             if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 input_path = os.path.join(folder_path, file_name)
#                 output_path = os.path.join(self.output_folder, file_name)
#                 self.crop_image(input_path, output_path, size)

#     def compress_images_in_folder(self, folder_path):
#         """Compress all images in a specified folder to the target size."""
#         for file_name in os.listdir(folder_path):
#             if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 input_path = os.path.join(folder_path, file_name)
#                 output_path = os.path.join(self.output_folder, file_name)
#                 self.compress_image(input_path, output_path)

#     def _center_crop(self, img, size):
#         """Center-crop the image to the specified size."""
#         width, height = img.size
#         target_width, target_height = size

#         left = (width - target_width) / 2
#         top = (height - target_height) / 2
#         right = (width + target_width) / 2
#         bottom = (height + target_height) / 2

#         return img.crop((left, top, right, bottom))

# def main():
#     parser = argparse.ArgumentParser(description='Image processing tool for cropping and compressing images.')
#     parser.add_argument('--mode', choices=['crop', 'compress'], required=True, help='Mode of operation: crop or compress.')
#     parser.add_argument('--is_folder', action='store_true', help='Specify if processing a folder. Otherwise, it processes a single image.')
#     parser.add_argument('--input_path', required=True, help='Path to the input image or folder.')
#     parser.add_argument('--output_folder', default='processed_images', help='Output folder for processed images (default: processed_images).')
#     parser.add_argument('--target_size_kb', type=int, default=100, help='Target size in KB for compression (default: 100).')

#     args = parser.parse_args()

#     editor = ImageEditor(target_size_kb=args.target_size_kb, output_folder=args.output_folder)

#     if args.is_folder:
#         if args.mode == 'compress':
#             editor.compress_images_in_folder(args.input_path)
#         elif args.mode == 'crop':
#             editor.crop_images_in_folder(args.input_path)
#     else:
#         output_path = os.path.join(args.output_folder, os.path.basename(args.input_path))
#         if args.mode == 'compress':
#             editor.compress_image(args.input_path, output_path)
#         elif args.mode == 'crop':
#             editor.crop_image(args.input_path, output_path)

# if __name__ == '__main__':
#     main()
