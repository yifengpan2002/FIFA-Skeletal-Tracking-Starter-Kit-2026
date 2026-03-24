import cv2
import os
from PIL import Image
from pathlib import Path


def extract_frames(video_path, output_folder, save_as_png=False):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is returned (end of video)
        
        # Save each frame as an image file
        if save_as_png:
            output_filename = os.path.join(output_folder, f'{frame_count:05d}.png')
        else:
            output_filename = os.path.join(output_folder, f'{frame_count:05d}.jpg')
        
        Image.fromarray(frame[..., ::-1]).save(output_filename, optimize=True)
        # cv2.imwrite(output_filename, frame)
        frame_count += 1

    cap.release()
    print(f'Done! {frame_count} frames extracted and saved in "{output_folder}".')

def process_folder(video_folder, output_root, save_as_png=False):

    video_folder = Path(video_folder)
    output_root = Path(output_root)

    video_files = list(video_folder.glob("*.mp4"))

    print(f"Found {len(video_files)} videos")

    for video in video_files:

        video_name = video.stem
        output_folder = output_root / video_name

        extract_frames(video, output_folder, save_as_png)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from videos")

    parser.add_argument("--video_folder", type=str, required=True,
                        help="Folder containing videos")

    parser.add_argument("--output_root", type=str, required=True,
                        help="Output folder for extracted frames")

    parser.add_argument("--png", action="store_true",
                        help="Save frames as PNG")

    args = parser.parse_args()

    process_folder(args.video_folder, args.output_root, args.png)

    # extract_frames(args.video_path, args.output_folder, args.png)