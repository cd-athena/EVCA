"""EVCA row `f` must describe the transition `f-1 -> f` (Phase 1.2 alignment).

The harness merges EVCA row f with the Low-Delay-P bits of encoded frame f, so an
off-by-one here would silently corrupt every frame-level correlation. This builds a
sequence with events at known indices and asserts the metrics fire on those exact
rows. The same probe was verified against x265 LDP bits at 1080p (see
validation/RESULTS.md, Gate 1): GT bits and EVCA metrics spike on identical indices.
"""
import numpy as np
import pytest

from tests.conftest import make_args, run_evca, write_raw_yuv
from validation.synthetic import make_base_texture

# Large enough that interior blocks outnumber the border ring: a global pan makes
# edge blocks reference pixels outside the frame, which leaves real residual there.
HEIGHT, WIDTH = 320, 448
PAN_START, CUT_AT, N_FRAMES = 4, 8, 12
PAN_PX = 4


@pytest.fixture(scope='module')
def event_yuv(tmp_path_factory):
    """Static, then a 4 px/frame pan from PAN_START, then a hard cut at CUT_AT."""
    canvas = make_base_texture(HEIGHT + 64, WIDTH + 64, seed=31)
    other = make_base_texture(HEIGHT, WIDTH, seed=32)
    frames = []
    for f in range(N_FRAMES):
        if f < PAN_START:
            frames.append(canvas[0:HEIGHT, 0:WIDTH])
        elif f < CUT_AT:
            off = (f - PAN_START + 1) * PAN_PX
            frames.append(canvas[0:HEIGHT, off:off + WIDTH])
        else:
            frames.append(other)
    path = tmp_path_factory.mktemp('align') / 'events.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    return str(path)


@pytest.fixture(scope='module')
def event_df(event_yuv, tmp_path_factory):
    out = tmp_path_factory.mktemp('align_csv') / 'events.csv'
    args = make_args(input=event_yuv, resolution=f'{WIDTH}x{HEIGHT}', csv=str(out),
                     motion_estimation=True, profile='full')
    return run_evca(args)


def test_static_prefix_is_quiet(event_df):
    """Frames 1..PAN_START-1 repeat the same picture: every temporal metric is ~zero.

    TC_MC keeps a bilinear-resampling floor (~1e-3) because even a zero MV field is
    applied through grid_sample; it is five orders below the cut-frame value.
    """
    quiet = event_df.iloc[1:PAN_START]
    for col in ['TC', 'TC_SAD', 'mean_mv_mag']:
        assert (quiet[col].abs() < 1e-6).all(), f'{col} nonzero on static frames'
    assert (quiet['TC_MC'].abs() < 1e-2).all()
    assert quiet['TC_MC'].max() < 1e-3 * event_df['TC_MC'].max()


def test_motion_onset_lands_on_first_moved_frame(event_df):
    """Row PAN_START is the first frame whose predecessor differs -> first nonzero."""
    assert event_df.loc[PAN_START - 1, 'mean_mv_mag'] == 0.0
    assert abs(event_df.loc[PAN_START, 'mean_mv_mag'] - PAN_PX) < 1e-6
    assert event_df.loc[PAN_START, 'TC'] > 0


def test_pan_is_motion_compensated(event_df):
    """A pure in-range pan must compensate away most of the raw temporal energy.

    The residual that remains sits in the border ring, whose blocks reference pixels
    outside the reference frame; interior-block exactness is asserted in
    test_synthetic_motion.test_mc_residual_near_zero_for_integer_translation.
    """
    pan = event_df.iloc[PAN_START:CUT_AT]
    assert (pan['mean_mv_mag'] == PAN_PX).all()
    assert (pan['TC_MC'] < 0.05 * pan['TC']).all(), 'MC residual not far below raw TC'
    assert (pan['intra_frac'] < 0.05).all()


def test_cut_spikes_on_cut_frame(event_df):
    """The hard cut must fire on row CUT_AT and nowhere else."""
    row = event_df.loc[CUT_AT]
    assert row['intra_frac'] > 0.9, 'intra gate did not fire across the cut'
    assert row['TC_MC'] == event_df['TC_MC'].max()
    assert row['TC_SAD'] == event_df['TC_SAD'].max()
    # after the cut the picture is static again
    after = event_df.iloc[CUT_AT + 1:]
    assert (after['TC_SAD'].abs() < 1e-6).all()
