import os
import random
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def split_video(filename, start_time_str, end_time_str, output_dir):
    clip = VideoFileClip(filename)
    duration = clip.duration

    start_time = time_to_seconds(start_time_str)
    end_time = time_to_seconds(end_time_str)

    if start_time >= duration:
        print("Start time melebihi durasi video.")
        return
    if end_time > duration:
        end_time = duration

    basename = os.path.basename(filename)

    output_path = os.path.join(os.path.dirname(filename), output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    random_number = random.randint(1000, 9999)
    output_file = os.path.join(output_path, f"{basename}_{random_number}.mp4")
    ffmpeg_extract_subclip(filename, start_time, end_time, targetname=output_file)

    print(f'Video berhasil dipotong dari {start_time_str} hingga {end_time_str} dan disimpan sebagai {output_file}')

if __name__ == "__main__":
    video_path = r"C:\Users\DOT\Web\RecordFiles\2024-11-06\192.168.1.10_01_20241106170840406.mp4"
    start_time = "02:40" # "33:00 " # format mm:ss
    end_time = "03:10"   # format mm:ss

    output_dir = "split_video"

    split_video(video_path, start_time, end_time, output_dir)