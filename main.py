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
    print('--heuristic               Search pattern: "diamond" (17-point, default), "diamond_dense" (21-point),')
    print('                          "diamond_axis" (13-point Iteration-4 plus, no diagonals), "square" (9-point).')
    print('--loader                  Select the I/O pipeline: "standard" (sequential reads, low memory) or "optimized" (memory-mapped, high throughput but may cause OOM error).')
    print('--bit_depth               Bit depth of the raw YUV video. Default: 8')
    print('--profile                 ME Profile. "fast" outputs spatial TC_SAD/MVC. "full" executes the heavy DCT to output true TC_MC.')
    print('--device                  Compute device: "auto" (CUDA > MPS > CPU), "cuda", "mps", or "cpu". Default: auto')
    print('--prefetch                Overlap GOP loading with compute via a background thread. 1=on (default), 0=off.')
    print("\nMotion estimation strategy (all defaults reproduce the Iteration-4 reference):")
    print('--me                      Search strategy: "pattern" (sparse diamond/square, default) or "hierarchical".')
    print('--me-subpel               Sub-pixel refinement: 0=integer (default), 1=half-pel, 2=quarter-pel.')
    print('--me-predictor            Candidate seeding: "none" (default) or "global" (phase-correlation global MV).')
    print('--me-lambda               MV-cost weight against the median predictor. Default 0 (off).')
    print('--me-merge                Enable the neighbour-MV re-evaluation pass. Default off.')
    print('--me-criterion            Block cost: "sad" (default) or "satd" (8x8 Hadamard).')
    print("\nMotion compensation strategy:")
    print('--mc                      "dense_smooth" (default), "dense", "block", or "obmc".')
    print('--mc-smooth               MV-field filter for dense modes: "gauss" (default), "median", "none".')
    print('--residual-dc             Keep the DC coefficient in the residual energy. Default off.')
    print('--gate                    Residual gating: "intra" (default, min(SC_MC, SC)) or "none".')
    print('--dct-impl                DCT backend: "matmul" (default, cached basis) or "torch_dct" (FFT reference).')
    print('--preset                  Apply a named flag bundle, e.g. "iter4". Explicit flags still win.')

# Named flag bundles. A preset only fills in flags the user did not pass explicitly,
# so an explicit flag always wins over the preset that would have set it.
PRESETS = {
    # The Iteration-4 reference configuration: half-resolution sparse-diamond search,
    # integer MVs, Gaussian-smoothed dense warp, intra-gated residual energy.
    'iter4': {
        'me': 'pattern', 'me_subpel': 0, 'me_predictor': 'none', 'me_lambda': 0.0,
        'me_merge': False, 'me_criterion': 'sad', 'heuristic': 'diamond_axis',
        'mc': 'dense_smooth', 'mc_smooth': 'gauss', 'residual_dc': False,
        'gate': 'intra',
    },
}


def _add_arguments(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """Declares every CLI flag. With `suppress`, unspecified flags are omitted from the
    namespace entirely, which is how preset application tells explicit flags apart."""
    def d(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument('-i', '--input', type=str, default=d('test.yuv'))
    parser.add_argument('-d', '--dir', type=str, default=d(None))
    parser.add_argument('-m', '--method', type=str, default=d('EVCA'))
    parser.add_argument('-t', '--transform', type=str, default=d('DCT'))
    parser.add_argument('-r', '--resolution', type=str, default=d('1920x1080'))
    parser.add_argument('-b', '--block_size', type=int, default=d(32))
    parser.add_argument('-f', '--frames', type=int, default=d(0))
    parser.add_argument('-c', '--csv', type=str, default=d('./csv/test.csv'))
    parser.add_argument('-g', '--gopsize', type=int, default=d(32))
    parser.add_argument('-p', '--pix_fmt', type=str, default=d('yuv420'))
    parser.add_argument('-s', '--sample_rate', type=int, default=d(1))
    parser.add_argument('-bi', '--block_info', type=int, default=d(0))
    parser.add_argument('-pi', '--plot_info', '-plot_info', type=int, nargs='?', const=1, default=d(0))
    parser.add_argument('-dp', '--dpi', type=int, default=d(100))
    parser.add_argument('-fi', '--filter', type=str, default=d('sobel'))
    parser.add_argument('-cf', '--colorfulness', action='store_true', default=d(False))
    parser.add_argument('-pm', '--plot_metrics', '-plot_metrics', type=int, nargs='?', const=1, default=d(0))
    parser.add_argument('-cc', '--chroma_complexity', action='store_true', default=d(False))
    parser.add_argument('-me', '--motion_estimation', action='store_true', default=d(False))
    # Kept in sync with libs.temporal_engine.SEARCH_PATTERNS by
    # tests/test_strategies.py::test_heuristic_choices_match_patterns; spelled out
    # here so `--help` does not have to import torch.
    parser.add_argument('--heuristic', type=str, default=d('diamond'),
                        choices=['diamond', 'diamond_axis', 'diamond_dense', 'square'])
    parser.add_argument('--loader', type=str, default=d('standard'), choices=['standard', 'optimized'])
    parser.add_argument('--bit_depth', type=int, default=d(8), choices=[8, 10, 12, 16])
    parser.add_argument('--profile', type=str, default=d('fast'), choices=['fast', 'full'])
    parser.add_argument('--device', type=str, default=d('auto'), choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--prefetch', type=int, default=d(1), choices=[0, 1])

    # --- Motion estimation strategy (Phase 2/3) ---
    parser.add_argument('--me', dest='me', type=str, default=d('pattern'),
                        choices=['pattern', 'hierarchical'])
    parser.add_argument('--me-subpel', dest='me_subpel', type=int, default=d(0), choices=[0, 1, 2])
    parser.add_argument('--me-predictor', dest='me_predictor', type=str, default=d('none'),
                        choices=['none', 'global'])
    parser.add_argument('--me-lambda', dest='me_lambda', type=float, default=d(0.0))
    parser.add_argument('--me-merge', dest='me_merge', action='store_true', default=d(False))
    parser.add_argument('--me-criterion', dest='me_criterion', type=str, default=d('sad'),
                        choices=['sad', 'satd'])

    # --- Motion compensation strategy (Phase 2/4) ---
    parser.add_argument('--mc', dest='mc', type=str, default=d('dense_smooth'),
                        choices=['dense_smooth', 'dense', 'block', 'obmc'])
    parser.add_argument('--mc-smooth', dest='mc_smooth', type=str, default=d('gauss'),
                        choices=['gauss', 'median', 'none'])
    parser.add_argument('--residual-dc', dest='residual_dc', action='store_true', default=d(False))
    parser.add_argument('--gate', dest='gate', type=str, default=d('intra'), choices=['intra', 'none'])
    parser.add_argument('--dct-impl', dest='dct_impl', type=str, default=d('matmul'),
                        choices=['matmul', 'torch_dct'])
    parser.add_argument('--preset', dest='preset', type=str, default=d(None), choices=sorted(PRESETS))


def get_parser_arguments(argv=None) -> argparse.Namespace:
    """Parses CLI arguments; pass an explicit argv list (e.g. []) for programmatic use."""
    parser = argparse.ArgumentParser(add_help=False, )
    _add_arguments(parser)
    args = parser.parse_args(argv)

    if getattr(args, 'preset', None):
        # Re-parse with suppressed defaults: only flags the user actually typed appear.
        probe = argparse.ArgumentParser(add_help=False)
        _add_arguments(probe, suppress=True)
        explicit = vars(probe.parse_args(argv))
        for dest, value in PRESETS[args.preset].items():
            if dest not in explicit:
                setattr(args, dest, value)
    return args


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
            if args.device == 'auto':
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    device = torch.device('mps')
                else:
                    device = torch.device('cpu')
            else:
                device = torch.device(args.device)
            print(f"Using device: {device.type}")
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
