import numpy as np
# from paper "Measuring colourfulness in natural images" by David Hasler and Sabine Süusstrunk
#
# M^(3) = sigma_rgyb + 0.3 * mu_rgyb
#
# where
# sigma_rgyb := sqrt(sigma^2_rg + sigma^2_yb)
# mu_rgyb := sqrt(mu^2_rg + mu^2_yb)
#
# with the color space:
# rg = R-G
# yb = 1/2 * (R+G) - B
#
# and the quantities:
# sigma_ab = sqrt(sigma^2_a + sigma^2_b)  : the trigonometric length of the standard deviation in ab space
# sigma_a: standard deviation along the a axis
# sigma_b: standard deviation along the b axis
#
# mu_ab: the distance of the centre of gravity in ab space to the neutral axis

def calculate_hasler_suesstrunk_colorfulness_yuv(plane_u: np.ndarray, plane_v: np.ndarray, bit_depth: int = 8) -> float:
    """
        Calculates the colorfulness using the Hasler & Süsstrunk metric directly from native YUV chroma planes.
        
        Avoids YUV-RGB color space conversion and chroma upsampling (e.g. 4:2:0 -> 4:4:4).
        Since U and V are inherently opponent color axes (blue-diff and red-diff), they (should)
        provide a good approximation of the Hasler & Süsstrunk metric natively.
        
    Args:
        plane_u (np.ndarray): The U (Cb) chroma plane (2D array).
        plane_v (np.ndarray): The V (Cr) chroma plane (2D array).
        bit_depth (int): The bit depth of the video (usually 8, 10, or 12).
                         Needed to determine the neutral chroma point.

    Returns:
        float: The approximated YUV colorfulness metric.
    """
    if plane_u.shape != plane_v.shape:
        raise ValueError(f"U and V planes must have identical dimensions. Got {plane_u.shape} and {plane_v.shape}")
    
    # neutral point for chroma: 2^(bit_depth - 1)
    # e.g. 128 for 8-bit, 512 for 10-bit
    neutral_point = 1 << (bit_depth - 1)
    
    
    # calculate variance (sigma^2 - faster since it omits an internal call to sqrt())
    var_u = np.var(plane_u)
    var_v = np.var(plane_v)
    
    # calculate means
    mean_u = np.mean(plane_u) - neutral_point
    mean_v = np.mean(plane_v) - neutral_point
    
    # trigonometric length of standard deviations
    sigma_uv = np.sqrt(var_u + var_v)
    
    # distance of center of gravity to neutral axis
    mu_uv = np.sqrt(mean_u**2 + mean_v**2)
    
    colorfulness_metric = sigma_uv + 0.3 * mu_uv
    
    return float(colorfulness_metric)
