"""
# File: Amit Tzadok
# Author: Amit Tzadok <amit.tzadok@icloud.com>
# Created: 2025-07-21 19:45:04
# Description: This script downloads a video from a URL, extracts its audio, converts the audio format, and transcribes the audio to text.
"""

import yt_dlp
from moviepy import VideoFileClip
import librosa
import soundfile as sf
import os

# --- Download video with merged audio ---
def download_video(url, output_path):
    try:
        # Set options for yt-dlp to ensure video and audio are included
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  # Ensure final output is mp4
            }],
        }
        
        # Download the video with audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            print(info)
            print(f"Downloading: {info['title']}")
            ydl.download([url])
            print(f"Download completed! File saved to {output_path}")
            return info['title']
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# --- Extract audio from video ---
def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

# --- Convert audio format ---
def convert_audio_format(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None)
    sf.write(output_path, audio, sr)

# --- Main function to orchestrate the process ---
def url2wav_main(video_url, output_path):
    # Download the video
    video_title = download_video(video_url, output_path)

    video_path= os.path.join(output_path, f"{video_title}.mp4")
    audio_path = os.path.join(output_path, f"{video_title}.mp3")
    wav_path = os.path.join(output_path, f"{video_title}.wav")
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist. Please check the download process.")
        return
    # Extract audio from the downloaded video
    extract_audio(video_path, audio_path)
    # Convert the extracted audio to the desired format
    convert_audio_format(audio_path, wav_path)

    print(f"Audio extracted and saved to: {wav_path}")

    return wav_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download a video, extract its audio, and convert the audio format.")

    parser.add_argument("--url", type=str, required=True, help="The URL of the video to download.")
    parser.add_argument("--path", type=str, default=None, help="The path to save the downloaded video and audio.")
    args = parser.parse_args()

    if (args.path is None):
        import platform
        if platform.system() == "Windows":
            downloads_dir = os.path.join(os.environ["USERPROFILE"], "Downloads")
        else:
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    else:
        downloads_dir = args.path

    os.makedirs(os.path.dirname(downloads_dir), exist_ok=True)
    url2wav_main(args.url, downloads_dir)
