from pathlib import Path
import shutil
import yt_dlp
import time
import subprocess
import os
import cv2

def download_vod(url, output_dir, start_sec, duration=480):
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'format': 'bestvideo[height<=1080]',
        'download_ranges': yt_dlp.utils.download_range_func(
            [], [[start_sec, start_sec + duration]]
        ),
        'force_keyframes_at_cuts': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return f"{output_dir}/{info['id']}.mp4"

def extract_draft_window(video_path, output_dir, duration=240, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-t", str(duration),
        "-vf", f"fps={fps}",
        f"{output_dir}/frame_%05d.jpg"
    ], check=True)

def list_formats(url):
    with yt_dlp.YoutubeDL({'listformats': True}) as ydl:
        ydl.extract_info(url, download=False)

#def process_vod(url, match_id, start_sec=180):
    # print(f"  Processing {match_id} (draft starts at {start_sec}s)")
 
    # video_output_dir = Path("backend/data/raw/vods") / match_id
    # frame_output_dir = Path("backend/data/raw/frames") / match_id

    # if frame_output_dir.exists() and any(frame_output_dir.iterdir()):
    #     print(f"  Frames already exist for {match_id}, skipping...")
    #     return str(frame_output_dir)

    # video_path = download_vod(url, str(video_output_dir), start_sec)
    # print(f"  Downloaded VOD to {video_path}")

    # extract_draft_window(video_path, str(frame_output_dir))
    # print(f"  Extracted draft frames to {frame_output_dir}")

    # time.sleep(1)
    # os.remove(video_path)
    # print(f"  Deleted VOD")

    # return str(frame_output_dir)

def get_stream_url(url, start_sec):
    """Get direct stream URL from YouTube without downloading"""
    ydl_opts = {
        'format': 'bestvideo[height<=1080]',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def extract_frames_from_stream(stream_url, output_dir, start_sec, duration=240, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(stream_url)
    
    # Seek to start position
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(source_fps / fps)  # only save every Nth frame
    
    end_msec = (start_sec + duration) * 1000
    frame_count = 0
    saved_count = 0

    print(f"  Streaming frames from {start_sec}s to {start_sec + duration}s...")

    while True:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_msec > end_msec:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_path = Path(output_dir) / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"  Saved {saved_count} frames to {output_dir}")

def process_vod(url, match_id, start_sec=180):
    print(f"  Processing {match_id} (draft starts at {start_sec}s)")

    frame_output_dir = Path("backend/data/raw/frames") / match_id

    if frame_output_dir.exists() and any(frame_output_dir.iterdir()):
        print(f"  Frames already exist for {match_id}, skipping...")
        return str(frame_output_dir)

    # Get stream URL (no download)
    print(f"  Getting stream URL...")
    stream_url = get_stream_url(url, start_sec)

    # Stream frames directly
    extract_frames_from_stream(stream_url, str(frame_output_dir), start_sec)
    print(f"  Done! No VOD saved to disk.")

    return str(frame_output_dir)
    