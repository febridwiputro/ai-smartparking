import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

# Fungsi untuk konversi menit:detik (mm:ss) menjadi detik total
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

    # Extract the filename without extension
    basename = os.path.basename(filename).split('.')[0]
    
    # Create output path
    output_path = os.path.join(os.path.dirname(filename), output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Set output file path
    output_file = os.path.join(output_path, f"{basename}_clip.mp4")

    # Extract the subclip
    ffmpeg_extract_subclip(filename, start_time, end_time, targetname=output_file)

    print(f'Video berhasil dipotong dari {start_time_str} hingga {end_time_str} dan disimpan sebagai {output_file}')

if __name__ == "__main__":
    # FLOOR 2 IN + 30s
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_2_IN.mp4"
    # start_time = "04:15"  # format mm:ss
    # end_time = "04:45"    # format mm:ss

    # FLOOR 3 IN + 01m
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_4_IN.mp4"
    # start_time = "04:45"  # format mm:ss
    # end_time = "05:30"    # format mm:ss

    # # FLOOR 4 IN + 01m:30s
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_4_IN.mp4"
    # start_time = "04:15"  # format mm:ss
    # end_time = "05:30"    # format mm:ss

    # FLOOR 5 IN + 02m:00s
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_4_IN.mp4"
    # start_time = "05:15" "03:45"  # format mm:ss
    # end_time = "05:30"    # format mm:ss

    # # FLOOR 5 OUT + 02m:30s
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_4_OUT.mp4"
    # start_time = "03:30"  # format mm:ss
    # end_time = "06:15"    # format mm:ss

    # FLOOR 4 OUT + 03m:00s
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_4_OUT.mp4"
    # start_time = "03:00"  # format mm:ss
    # end_time = "06:15"    # format mm:ss

    # FLOOR 3 OUT + 03m:30s
    # video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_3_OUT.mp4"
    # start_time = "03:35" # "06:35" # format mm:ss
    # end_time = "06:50"    # format mm:ss

    # FLOOR 2 OUT + 04m:00s
    video_path = r"D:\engine\smart_parking\dataset\cctv\2024-10-04\LT_2_OUT.mp4"
    start_time = "03:15" # "07:15"  # format mm:ss
    end_time = "07:28"    # format mm:ss

    output_dir = "split_video"

    split_video(video_path, start_time, end_time, output_dir)