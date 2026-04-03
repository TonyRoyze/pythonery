import os
from moviepy.editor import AudioFileClip, concatenate_audioclips

def merge_audios(audio_paths, output_path):

    for path in audio_paths:
        if not os.path.exists(path):
            print(f"Error: Input file not found: {path}")
            return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clips = []
    try:
        print("Loading audio clips...")
        for path in audio_paths:
            clip = AudioFileClip(path)
            clips.append(clip)

        print(f"Concatenating {len(clips)} clips...")
        final_clip = concatenate_audioclips(clips)

        print(f"Writing output to: {output_path}")
        final_clip.write_audiofile(output_path, codec='pcm_s16le')  # ✅ FIXED

        print("Done!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        for clip in clips:
            clip.close()

def main():

    # Manually specify the files here
    base_path = '/Users/viduragunawardana/Code/DataScience/Python/Scripts/Video Merging/demo/'
    audio_paths = [ base_path + f"{i}.wav" for i in range(0, 28) ]
    output_path = base_path + "demo.wav"

    merge_audios(audio_paths, output_path)

if __name__ == "__main__":
    main()
