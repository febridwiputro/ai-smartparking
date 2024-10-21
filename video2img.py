import cv2
import os

def split_video_to_images(video_path, num_split, output_dir='output_images'):
    """
    Splits a video or videos in a folder into images.

    Parameters:
    - video_path: str, path to the input video file or folder containing videos.
    - num_split: int, the number of images to save from each video.
    - output_dir: str, directory to save the output images.
    """

    if os.path.isfile(video_path):
        # Process a single video
        process_video(video_path, num_split, output_dir)
    elif os.path.isdir(video_path):
        # Process all videos in a folder
        for file in os.listdir(video_path):
            if file.endswith(('.mp4', '.avi', '.mkv')):  # Add more extensions if needed
                video_file = os.path.join(video_path, file)
                process_video(video_file, num_split, output_dir)
    else:
        print(f"Error: '{video_path}' is neither a valid file nor a folder.")

def process_video(video_file, num_split, output_dir):
    """
    Processes a single video and splits it into images.

    Parameters:
    - video_file: str, path to the video file.
    - num_split: int, the number of images to save.
    - output_dir: str, directory to save the output images.
    """
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_file}'.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_split)  # Avoid division by zero
    base_name = os.path.splitext(os.path.basename(video_file))[0]

    video_output_dir = os.path.join(output_dir, base_name)  # Separate folder for each video
    os.makedirs(video_output_dir, exist_ok=True)

    saved_frames = 0
    for i in range(num_split):
        frame_index = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if success:
            output_file = os.path.join(video_output_dir, f'{base_name}_frame_{saved_frames:03d}.jpg')
            cv2.imwrite(output_file, frame)
            print(f"Saved: {output_file}")
            saved_frames += 1
        else:
            print(f"Error: Could not read frame at index {frame_index}.")
            break

    cap.release()
    print(f"Processing complete for video: {video_file}")

# Example usage

# Single video input
# video_input = r"C:\Users\DOT\Videos\vlc-record-2024-09-30-09h52m30s-lantai_5_in_7_00.mp4"

# Folder containing multiple videos
video_input = r"C:\Users\DOT\Videos"

num_split_images = 10
split_video_to_images(video_input, num_split_images)
