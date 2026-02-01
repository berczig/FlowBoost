import os
import re
import ffmpeg
from pathlib import Path

# Function to extract bstart and bend from the file name
def extract_bstart_bend(filename):
    match = re.match(r'bstart=(\d+\.\d+),bend=(\d+\.\d+)\.mp4', filename)
    if match:
        bstart = float(match.group(1))
        bend = float(match.group(2))
        return bstart, bend
    return None, None

# Function to process the videos and combine them into a grid layout
def create_video_grid(input_folder, output_filename):
    # Get all .mp4 files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    if not video_files:
        raise Exception("No .mp4 files found in the input folder.")
    
    # Prepare inputs and labels for each video
    inputs = []
    overlay_filters = []
    for idx, video_file in enumerate(video_files):
        bstart, bend = extract_bstart_bend(video_file)
        if bstart is None or bend is None:
            print(f"Skipping invalid file {video_file}")
            continue
        
        # Create the input for the video file
        video_path = os.path.join(input_folder, video_file)
        inputs.append(ffmpeg.input(video_path))
        
        # Scale video down (e.g., scale to 320x240)
        scale_filter = f"scale=320:240"
        
        # Add text label (position it at the top of the video)
        text_filter = f"drawtext=text='bstart={bstart:.7f}, bend={bend:.7f}':x=10:y=10:fontsize=24:fontcolor=white"
        
        # Combine scale and drawtext filter
        overlay_filters.append(f"[{idx}:v] {scale_filter}, {text_filter} [v{idx}]")
    
    # Generate filtergraph to combine all videos into a grid
    # Assuming a square layout, e.g., 2x2 grid for 4 videos
    filtergraph = ";".join(overlay_filters)
    video_concat = "".join([f"[v{i}]" for i in range(len(video_files))])
    
    # Final output video filtergraph (arranging into a 2x2 grid layout)
    layout = f"x=0:y=0,scale=320x240[v0];x=320:y=0,scale=320x240[v1];x=0:y=240,scale=320x240[v2];x=320:y=240,scale=320x240[v3]"
    
    # Run ffmpeg to combine videos into a single output file
    output_path = os.path.join(input_folder, output_filename)
    ffmpeg.input(inputs).output(output_path, vcodec='libx264', acodec='aac').run()


input_folder = Path(__file__).parent
output_filename = 'combined_output.mp4'

# Create the combined video
create_video_grid(input_folder, output_filename)
