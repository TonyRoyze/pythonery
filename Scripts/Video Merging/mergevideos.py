import os

from moviepy.editor import VideoFileClip, concatenate_videoclips

def merge_videos(video_paths, output_path, method='compose'):
    """
    Merges a list of video clips into a single video file.

    Args:
        video_paths (list): List of paths to the input video files.
        output_path (str): Path where the merged video will be saved.
        method (str): Concatenation method. 'compose' is more robust but slower.
                      'chain' is faster but requires same resolution/codecs.
    """
    
    # Validate input files
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Error: Input file not found: {path}")
            return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return

    clips = []
    try:
        print("Loading video clips...")
        for path in video_paths:
            clip = VideoFileClip(path)
            clips.append(clip)
            print(f"Loaded: {path}")

        print(f"Concatenating {len(clips)} clips...")
        # method='compose' handles different resolutions/formats better
        final_clip = concatenate_videoclips(clips, method=method)

        print(f"Writing output to: {output_path}")
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        print("Done!")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close clips to release resources
        for clip in clips:
            try:
                clip.close()
            except:
                pass

def main():

    # Manually specify the files here
    base_path = "/Users/viduragunawardana/Code/DataScience/Python/Ridge Regression/media/videos/scaling_animation/1080p30/partial_movie_files/ScalingAnimation/"
    video_paths = [ base_path + f"clip{i}.mp4" for i in range(17, 24) ]
    output_path = base_path + "section3.mp4"
    method = 'chain' # or 'compose'

    merge_videos(video_paths, output_path, method)

if __name__ == "__main__":
    main()
