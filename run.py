import argparse
import cv2
import glob

from src.timeline import Timeline


def show_saved_images():
    for path in glob.glob("saved_images/**.png"):
        img = cv2.imread(path)
        cv2.imshow("frame", img)
        cv2.waitKey(30)

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
        '--read-cached',
        action='store_true',
        help='reads images from precalculated results'
    )
    
    args = parser.parse_args()

    if not args.read_cached:
        raw_start()
    else:
        show_saved_images()
