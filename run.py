import argparse
import cv2
import glob

from src.timeline import Timeline


def show_saved_images():
    files = glob.glob("saved_images/**.png")
    if len(files) > 0:
        for path in files:
            img = cv2.imread(path)
            cv2.imshow("frame", img)
            cv2.waitKey(30)
    else:
        print("No images in saved_images folder to read. Starting from scratch...")
        raw_start()

def raw_start():
    print("Raw start...")
    timeline = Timeline()
    timeline.interpolate_frames_and_save()
    show_saved_images()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='============= CLI for Visual geometry project ============='
    )
    
    parser.add_argument(
        '--raw-start',
        action='store_true',
        help='starts to recover 3d scene and injecting cube into video'
    )
    
    args = parser.parse_args()

    if args.raw_start:
        raw_start()
    else:
        show_saved_images()
