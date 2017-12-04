import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='============= CLI for Visual geometry project =============',
        epilog='============= Innsbruck university winter 2017 ==============\n'
    )
    
    # camera recalibatrion
    parser.add_argument(
        '--recalibrate', '-r',
        nargs=1,
        type=str,
        default='videos/checkerboard_calibration.mp4',
        help='recalibrate camera using checkerboard pattern (default: %(default)s)',
        metavar='[path to checkerboard video]'
    )
    
    # epipolar lines showcase
    parser.add_argument(
        '--epipolarlines', '-e',
        action='store_true',
        help='run a video and draw epipolar lines on it'
    )
    
    args = parser.parse_args()



    print("YO!")