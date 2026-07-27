print("Importing libraries...")
import argparse
import glob
import sys
import time
from pathlib import Path

print("Libraries imported successfully.\n\n")


def check_existence(input_args):
    files_list = []
    if input_args.dir:
        files_list = [file for file in glob.glob('{}/*.yuv'.format(input_args.dir))]
    elif input_args.input:
        if Path(input_args.input).is_file():
            files_list = [input_args.input]
    else:
        return files_list, False

    return files_list, True


def print_custom_help():
    print("EVCA:    Enhanced Video Complexity Analyzer v1.0")
    print("Usage:   python3 main.py [options]")
    print("\nOptions:")
    print("-h /--help                Show this help text and exit.")
    print("-m /--method              Feature extraction method. Default is EVCA. [VCA, EVCA, SITI] ")
    print("-t /--transform           Discrete transform method. Default is DCT. [DCT, DWT, DCT_B]. Note: DCT_B only compatible with block size 32")
    print("-fi/--filter              Edge detection filter. Default is sobel filter. [sobel, canny] ")
    print("-i /--input               Raw YUV input file name.")
    print("-d /--dir                 Directory to multiple yuv files.")
    print("-r /--resolution          Set the resolution [w]x[h]. Default is 1920x1080.")
    print("-b /--block_size          Set the block size. Default is 32 and must be a multiple of 4.")
    print("-f /--frames              Maximum number of frames for features extraction. 0 for all frames.")
    print("-g /--gopsize             The number of frames that is processed simultaneously. "
          "Default is 32 and should be greater equal 2.")
    print("-p /--pix_fmt             yuv format. Default is yuv420. [yuv420, yuv444] ")
    print("-s /--sample_rate         Frame subsampling. Default is 1 ")
    print("-c /--csv                 Name of csv to write features. Default is ./csv/test.csv")
    print("-bi/--block_info          Write block level features into a csv. Default is disabled")
    print("-pi/--plot_info           Plot per frame features for each frame. Default is disabled")
    print("-dp/--dpi                 Image quality of the saved output. Default is 100.")
    print("-cf /--colorfulness       Enable Hasler & Süsstrunk M^(3) colorfulness analysis.")
    print("-pm/--plot_metrics        Plot per frame metrics over time. Default is disabled")   
    print("-cc/--chroma_complexity   Enable Chroma (U,V) Complexity calculation.")
    print("-me/--motion_estimation   Enable Block-Based Motion Estimation (Block-ME)")
    print('--heuristic               Search pattern geometry: "diamond" (13-point) or "square" (9-point).')
    print('--loader                  Select the I/O pipeline: "standard" (sequential reads, low memory) or "optimized" (memory-mapped, high throughput but may cause OOM error).')
    print('--bit_depth               Bit depth of the raw YUV video. Default: 8')
    print('--profile                 ME Profile. "fast" outputs spatial TC_SAD/MVC. "full" executes the heavy DCT to output true TC_MC.')

def get_parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, )
    parser.add_argument('-i', '--input', type=str, default='test.yuv')
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-m', '--method', type=str, default='EVCA')
    parser.add_argument('-t', '--transform', type=str, default='DCT')
    parser.add_argument('-r', '--resolution', type=str, default='1920x1080')
    parser.add_argument('-b', '--block_size', type=int, default='32')
    parser.add_argument('-f', '--frames', type=int, default='0')
    parser.add_argument('-c', '--csv', type=str, default='./csv/test.csv')
    parser.add_argument('-g', '--gopsize', type=int, default='32')
    parser.add_argument('-p', '--pix_fmt', type=str, default='yuv420')
    parser.add_argument('-s', '--sample_rate', type=int, default='1')
    parser.add_argument('-bi', '--block_info', type=int, default='0')
    parser.add_argument('-pi', '--plot_info', type=int, default='0')
    parser.add_argument('-dp', '--dpi', type=int, default='100')
    parser.add_argument('-fi', '--filter', type=str, default='sobel')
    parser.add_argument('-cf','--colorfulness', action='store_true')
    parser.add_argument('-pm', '--plot_metrics', type=int, default='0')
    parser.add_argument('-cc', '--chroma_complexity', action='store_true')
    parser.add_argument('-me', '--motion_estimation', action='store_true')
    parser.add_argument('--heuristic', type=str, default='diamond', choices=['diamond', 'square'])
    parser.add_argument('--loader', type=str, default='standard', choices=['standard', 'optimized'])
    parser.add_argument('--bit_depth', type=int, default=8, choices=[8, 10, 12, 16])
    parser.add_argument('--profile', type=str, default='fast', choices=['fast', 'full'])
    return parser.parse_args()


def main():
    # Check if the help option is explicitly provided
    if '-h' in sys.argv or '--help' in sys.argv:
        print_custom_help()
    else:

        args = get_parser_arguments()

        import torch
        from libs.EVCA import EVCA
        from libs.SITI import SITI
        print("EVCA: Enhanced Video Complexity Analyzer v1.1.")

        # Check input file(s) existence.
        input_list, success = check_existence(args)
        if success:
            print("Start to extract features...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            t1 = time.time()
            if args.method == 'EVCA':
                EVCA(args, input_list, device)
            elif args.method == 'VCA':
                EVCA(args, input_list, device)
            elif args.method == 'SITI':
                SITI(args, input_list, device)
            else:
                print('Unsupported method.')
            t2 = time.time()
            print(f'Feature extraction completed in {t2 - t1:.2f} seconds.')
        else:
            print('Input file or directory is not specified.')


if __name__ == "__main__":
    main()
