"""EVCA - Enhanced Video Complexity Analyzer: command-line entry point."""
import argparse
import glob
import sys
import time
from pathlib import Path

__version__ = '2.0'

EPILOG = """\
examples:
  # Baseline spatial/temporal complexity over a whole sequence
  python main.py -i input.yuv -r 1920x1080 -c ./csv/out.csv

  # Full feature set: motion estimation and compensated residual, chroma, colorfulness
  python main.py -i input.yuv -r 1920x1080 -me --profile full -cc -cf -c ./csv/out.csv

  # Batch a directory of 10-bit sequences using the high-throughput loader
  python main.py -d ./sequences -r 3840x2160 --bit_depth 10 --loader optimized

Input is raw, uncompressed YUV: --resolution, --pix_fmt and --bit_depth must match
the file, since nothing in the container describes how to interpret the bytes.
"""


def check_existence(input_args):
    """Resolves the inputs to process. Returns (files, ok); ok is False when there is
    nothing to do, in which case the reason has already been reported."""
    if input_args.dir:
        # Sorted so a --dir run processes files in a reproducible order.
        files = sorted(glob.glob(f'{input_args.dir}/*.yuv'))
        if not files:
            print(f'No .yuv files found in directory: {input_args.dir}')
        return files, bool(files)
    if input_args.input:
        if Path(input_args.input).is_file():
            return [input_args.input], True
        print(f'Input file not found: {input_args.input}')
        return [], False
    print('No input specified: pass --input FILE or --dir DIR.')
    return [], False


def positive_int(value: str) -> int:
    """argparse `type` for options that must be at least 1."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {parsed}')
    return parsed


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Appends each option's default to its help text, and leaves the epilog and the
    group descriptions laid out as written.

    On/off switches are exempt from the default suffix: "(default: False)" on a flag
    whose only purpose is to turn something on carries no information.
    """

    def _get_help_string(self, action):
        if action.nargs == 0:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser.

    Help text lives on the arguments themselves, so `--help` is generated from the
    parser and cannot drift from what the tool actually accepts.
    """
    parser = argparse.ArgumentParser(
        description=f'EVCA: Enhanced Video Complexity Analyzer v{__version__}',
        epilog=EPILOG,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'EVCA {__version__}')

    src = parser.add_argument_group('input')
    src.add_argument('-i', '--input', default='test.yuv', metavar='FILE',
                     help='raw YUV input file')
    src.add_argument('-d', '--dir', metavar='DIR',
                     help='directory of .yuv files to process in turn; takes precedence '
                          'over --input')
    src.add_argument('-r', '--resolution', default='1920x1080', metavar='WxH',
                     help='source resolution')
    src.add_argument('-p', '--pix_fmt', default='yuv420', choices=['yuv420', 'yuv444'],
                     help='chroma subsampling of the input')
    src.add_argument('--bit_depth', type=int, default=8, choices=[8, 10, 12, 16],
                     metavar='BITS',
                     help='bit depth of the input; metrics are rescaled to an 8-bit '
                          'equivalent so runs at different depths stay comparable')
    src.add_argument('-f', '--frames', type=int, default=0, metavar='N',
                     help='maximum frames to analyze; 0 means all of them')
    src.add_argument('-s', '--sample_rate', type=positive_int, default=1, metavar='N',
                     help='analyze every Nth frame')

    ana = parser.add_argument_group('analysis')
    ana.add_argument('-m', '--method', default='EVCA', choices=['EVCA', 'VCA', 'SITI'],
                     help='feature extraction method')
    ana.add_argument('-t', '--transform', default='DCT', choices=['DCT', 'DWT', 'DCT_B'],
                     help='block transform; DCT_B requires --block_size 32')
    ana.add_argument('-b', '--block_size', type=positive_int, default=32, metavar='N',
                     help='block edge length in pixels; should be a multiple of 4')
    ana.add_argument('-g', '--gopsize', type=positive_int, default=32, metavar='N',
                     help='frames transformed per batch; larger values use more VRAM')
    ana.add_argument('-fi', '--filter', default='sobel', choices=['sobel', 'canny'],
                     help='edge filter for --method SITI; only sobel is implemented')

    feat = parser.add_argument_group('features')
    feat.add_argument('-cc', '--chroma_complexity', action='store_true',
                      help='also compute per-frame U and V complexity (SC_u, SC_v)')
    feat.add_argument('-cf', '--colorfulness', action='store_true',
                      help='also compute Hasler & Suesstrunk M^3 colorfulness from the '
                           'chroma planes')
    feat.add_argument('-me', '--motion_estimation', action='store_true',
                      help='enable block-based motion estimation, adding MVC, TC_SAD '
                           'and mean_mv_mag')
    feat.add_argument('--profile', default='fast', choices=['fast', 'full'],
                      help='"fast" runs the motion search only; "full" additionally '
                           'transforms the motion-compensated residual to produce TC_MC '
                           'and intra_frac, at roughly half the throughput')

    me = parser.add_argument_group(
        'motion estimation',
        'Candidates per block are scored by SAD against whole-frame shifts of the\n'
        'reference. --me-offset is always in full-resolution pixels; --temporal-pool\n'
        'sets the grid those pixels are quantised to, trading vector granularity for\n'
        'a search that costs four times less per candidate at each step.')
    me.add_argument('--heuristic', default='diamond',
                    choices=['diamond', 'square', 'dense'],
                    help='candidate placement: "diamond" puts four neighbours on the '
                         'axes, "square" on the diagonals, "dense" fills the whole '
                         'square so reach and granularity are independent')
    me.add_argument('--me-offset', dest='me_offset', type=positive_int, default=2,
                    metavar='PIXELS',
                    help='reach of the pattern in full-resolution pixels; motion beyond '
                         'it cannot be tracked')
    me.add_argument('--temporal-pool', dest='temporal_pool', type=int, default=1,
                    choices=[1, 2, 4], metavar='N',
                    help='box-filter the temporal path down by N before searching and '
                         'before building the motion-compensated residual; 1 reproduces '
                         'full-resolution behaviour exactly')
    me.add_argument('--me-pool', dest='me_pool', type=int, default=None,
                    choices=[1, 2, 4, 8], metavar='N',
                    help='pooling factor for the motion search alone; defaults to '
                         '--temporal-pool, and overriding it decouples the search from '
                         'the residual path')

    mc = parser.add_argument_group(
        'motion compensation',
        'Applies with --motion_estimation --profile full. The compensated residual\n'
        "energy is always capped at the block's own intra energy, modelling an\n"
        'encoder per-block inter/intra decision; intra_frac reports how often it binds.')
    mc.add_argument('--mc', default='dense_smooth',
                    choices=['dense_smooth', 'dense', 'block', 'obmc'],
                    help='how the block motion field becomes a pixel warp: bilinear '
                         'upsample of a filtered field, of an unfiltered field, '
                         'piecewise-constant blocks, or overlapped blocks')
    mc.add_argument('--mc-smooth', dest='mc_smooth', default='gauss',
                    choices=['gauss', 'median', 'none'],
                    help='motion-field filter for the dense --mc modes; "median" is a '
                         'vector median, so it never invents a vector')
    mc.add_argument('--residual-dc', dest='residual_dc', action='store_true',
                    help='keep the DC coefficient in the residual energy; for a '
                         "compensated residual DC is the block's mean prediction error "
                         "and costs real rate, unlike an intra block's DC")

    perf = parser.add_argument_group('performance')
    perf.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                      help='compute device; "auto" prefers CUDA, then MPS, then CPU')
    perf.add_argument('--loader', default='standard', choices=['standard', 'optimized'],
                      help='I/O pipeline: "standard" reads sequentially at low memory, '
                           '"optimized" memory-maps for higher throughput but can '
                           'exhaust RAM on very large files')
    perf.add_argument('--prefetch', type=int, default=1, choices=[0, 1],
                      help='overlap the next batch load with the current batch compute '
                           'on a background thread')

    out = parser.add_argument_group('output')
    out.add_argument('-c', '--csv', default='./csv/test.csv', metavar='FILE',
                     help='destination CSV for per-frame features; a <FILE>.meta.json '
                          'sidecar records the git SHA and full argument set')
    out.add_argument('-bi', '--block_info', type=int, nargs='?', const=1, default=0,
                     metavar='0|1',
                     help='also write per-block features to <csv>_<METRIC>_blocks.csv')
    out.add_argument('-pi', '--plot_info', '-plot_info', type=int, nargs='?', const=1,
                     default=0, metavar='0|1',
                     help='plot per-block features for every frame into ./png/; '
                          'implies --block_info')
    out.add_argument('-pm', '--plot_metrics', '-plot_metrics', type=int, nargs='?',
                     const=1, default=0, metavar='0|1',
                     help='plot per-frame metrics over time into ./png/frame_metrics/')
    out.add_argument('-dp', '--dpi', type=positive_int, default=100, metavar='N',
                     help='resolution of saved plots')

    return parser


def get_parser_arguments(argv=None) -> argparse.Namespace:
    """Parses CLI arguments; pass an explicit argv list (e.g. []) for programmatic use."""
    return build_parser().parse_args(argv)


def select_device(name: str):
    """Resolves --device to a torch device, preferring CUDA then MPS for "auto"."""
    import torch
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    args = get_parser_arguments()

    input_list, success = check_existence(args)
    if not success:
        return 1

    print(f'EVCA: Enhanced Video Complexity Analyzer v{__version__}')
    # torch is imported here, not at module scope, so --help and --version stay instant.
    print('Loading PyTorch...', flush=True)
    from libs.EVCA import EVCA
    from libs.SITI import SITI

    device = select_device(args.device)
    print(f'Using device: {device.type}')
    print('Start to extract features...')

    t1 = time.time()
    if args.method == 'SITI':
        SITI(args, input_list, device)
    else:
        EVCA(args, input_list, device)     # handles both EVCA and VCA
    t2 = time.time()
    print(f'Feature extraction completed in {t2 - t1:.2f} seconds.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
