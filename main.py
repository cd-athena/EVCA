import os
import sys
import time
import argparse




def print_custom_help():
    print("EVCA:    Enhanced Video Complexity Analyzer v1.0")
    print("Usage:   python3 main.py [options]")
    print("\nOptions:")
    print("-h /--help                Show this help text and exit.")
    print("-m /--method              Feature extraction method. Default is EVCA. [VCA, EVCA, SITI] ")
    print("-fi/--filter              Edge detection filter. Default is Sobel filter. [sobel, canny] ")
    print("-i /--input               Raw YUV input file name.")
    print("-r /--resolution          Set the resolution [w]x[h]. Default is 1920x1080.")
    print("-b /--block_size          Set the block size. Default is 32. [8, 16, 32]")
    print("-f /--frames              Maximum number of frames for features extraction.")
    print("-p /--pix_fmt             yuv format. Default is yuv420. [yuv420, yuv444] ")
    print("-s /--sample_rate         Frame subsampling. Default is 1 ")
    print("-c /--csv                 Name of csv to write features. Default is ./csv/test.csv")
    print("-bi/--block_info          Write block level features into a csv. Default is disabled")
    print("-pi/--plot_info           Plot per frame features. Default is disabled")
    print("-dp/--dpi                 Image quality of the saved output. Default is 100.")
def main():
    # Check if the help option is explicitly provided
    if '-h' in sys.argv or '--help' in sys.argv:
        print_custom_help()
    else:
        
        parser = argparse.ArgumentParser(add_help=False,)
        parser.add_argument('-i',  '--input', type=str, default='./0001.yuv')
        parser.add_argument('-m',  '--method', type=str, default='EVCA')
        parser.add_argument('-r',  '--resolution', type=str, default='1920x1080')
        parser.add_argument('-b',  '--block_size', type=int, default='32')
        parser.add_argument('-f',  '--frames', type=int, default='30')
        parser.add_argument('-c',  '--csv', type=str, default='./csv/test.csv')
        parser.add_argument('-p',  '--pix_fmt', type=str, default='yuv420')
        parser.add_argument('-s',  '--sample_rate', type=int, default='1') 
        parser.add_argument('-bi', '--block_info', type=int, default='0')
        parser.add_argument('-pi', '--plot_info', type=int, default='0')
        parser.add_argument('-dp', '--dpi', type=int, default='100')
        parser.add_argument('-fi', '--filter', type=str, default='sobel')
        
        args = parser.parse_args()
        print("Importing libraries...")
        import torch 
        import torch_dct as dct
        from libs.EVCA import EVCA
        from libs.SITI import SITI
        print("Libraries imported successfully.\n\n")
        # Your main script logic goes here
        print("EVCA: Enhanced Video Complexity Analyzer v1.0.")
        print("Start to extract features...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t1 = time.time()
        if args.method == 'EVCA':
            EVCA(args,device)
        elif args.method == 'VCA':
            EVCA(args,device)    
        elif args.method == 'SITI':
            SITI(args,device)
        else:
            print('Unsupported method.')
        t2 = time.time()
        print(f'Feature extraction completed in {t2-t1:.2f} seconds.')
        
if __name__ == "__main__":
    main()
