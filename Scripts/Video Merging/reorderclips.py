import os
import re

def reorder_clips(list_file_path):
    """
    Reads a partial movie file list (FFMPEG format) and renames the referenced files
    to clip1, clip2, etc. in their respective directories.
    """
    if not os.path.isfile(list_file_path):
        print(f"List file not found: {list_file_path}")
        return

    with open(list_file_path, 'r') as f:
        lines = f.readlines()

    # Filter lines that start with "file "
    file_lines = [line.strip() for line in lines if line.strip().startswith("file ")]

    for i, line in enumerate(file_lines):
        # Format is: file 'file:/absolute/path/to/video.mp4'
        # We need to extract the path inside the single quotes
        
        # Regex to extract content between single quotes
        match = re.search(r"'([^']*)'", line)
        if not match:
            print(f"Could not parse line: {line}")
            continue
            
        full_path_str = match.group(1)
        
        # Remove file: prefix if present
        if full_path_str.startswith("file:"):
            full_path_str = full_path_str[5:]
            
        original_filepath = full_path_str
        
        if not os.path.exists(original_filepath):
            print(f"File not found: {original_filepath}")
            continue
            
        directory = os.path.dirname(original_filepath)
        extension = os.path.splitext(original_filepath)[1]
        
        new_filename = f"clip{i+1}{extension}"
        new_filepath = os.path.join(directory, new_filename)
        
        if original_filepath == new_filepath:
            print(f"File is already named {new_filename}")
            continue
            
        try:
            os.rename(original_filepath, new_filepath)
            print(f"Renamed '{original_filepath}' to '{new_filename}'")
        except OSError as e:
            print(f"Error renaming '{original_filepath}': {e}")

if __name__ == "__main__":
    # Path to the specific list file provided by the user
    list_file = "/Users/viduragunawardana/Code/DataScience/Python/Ridge Regression/media/videos/scaling_animation/1080p30/partial_movie_files/ScalingAnimation/partial_movie_file_list.txt"
    reorder_clips(list_file)