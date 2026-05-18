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

def calculate_hasler_suesstrunk_colorfulness(frame_rgb: np.ndarray) -> float:
    """
    Calculates the colorfulness of an RGB frame using the Hasler & Süsstrunk metric.
    
    This function leverages an efficient sRGB opponent space to estimate human-perceived 
    colorfulness without requiring expensive non-linear color space conversions.
    
    Args:
        frame_rgb (np.ndarray): The input video frame of shape (H, W, 3).
                                Expected color channel order is RGB
                                Supports both uint8 (SDR) and uint16 (HDR) inputs
    
    Returns:
        float: The calculated M^(3) colorfulness metric. Higher values indicate higher colorfulness.
               Returns 0.0 if the frame is strictly grayscale.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"Expected a 3D array with 3 channels (RGB), got shape {frame_rgb.shape}")
    
    # upcast to float32 to prevent integer underflow during opponent channel subtraction &
    # flatten the spatial dimensions (H*W) to allow 1D numpy array optimizations for std and mean calcs
    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)
    
    # slice channels (readability & vectorized memory access)
    R = pixels[:, 0]
    G = pixels[:, 1]
    B = pixels[:, 2]
    
    # sRGB opponent space representations
    # rg: red-green axis
    rg = R - G
    # yb: yellow-blue axis
    yb = 0.5 * (R + G) - B
    
    # standard deviations of the opponent axes
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    
    # mean values of the opponent axes
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    
    # calculate sigma_rgyb (trigonometric length of standard deviation in opponent space)
    sigma_rgyb = np.sqrt(std_rg**2 + std_yb**2)
    
    # calculate mu_rgyb (distance of center of gravity to neutral axis)
    mu_rgyb = np.sqrt(mean_rg**2 + mean_yb**2)
    
    # calculate final metric M^(3)
    colorfulness_metric = sigma_rgyb + 0.3 * mu_rgyb
    
    return float(colorfulness_metric)


def analyze_video_colorfulness(video_frames: list) -> dict:
    """
    utility function to process a list of frames and return temporal colorfulness statistics.
    
    Args:
        video_frames (list): List of RGB frames (np.ndarray)
    
    Returns:
        dict: Statistical summary of video's colorfulness
    """
    if not video_frames:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    
    metrics = [calculate_hasler_suesstrunk_colorfulness(f) for f in video_frames]
    
    return {
        "mean_colorfulness": float(np.mean(metrics)),
        "std_colorfulness": float(np.std(metrics)),
        "max_colorfulness": float(np.max(metrics)),
        "min_colorfulness": float(np.min(metrics))
    }

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
    
    # ravel() creates a flattened view, which is faster than statistical ops
    # upcasting to float32 is required for centering shift to avoid underflow/overflow
    u_flat = plane_u.ravel().astype(np.float32)
    v_flat = plane_v.ravel().astype(np.float32)
    
    # center the chroma values around 0 (true neutral)
    u_centered = u_flat - neutral_point
    v_centered = v_flat - neutral_point
    
    # calculate variance (sigma^2 - faster since it omits an internal call to sqrt())
    var_u = np.var(u_centered)
    var_v = np.var(v_centered)
    
    # calculate means
    mean_u = np.mean(u_centered)
    mean_v = np.mean(v_centered)
    
    # trigonometric length of standard deviations
    sigma_uv = np.sqrt(var_u + var_v)
    
    # distance of center of gravity to neutral axis
    mu_uv = np.sqrt(mean_u**2 + mean_v**2)
    
    colorfulness_metric = sigma_uv + 0.3 * mu_uv
    
    return float(colorfulness_metric)
    


def analyze_yuv_video_colorfulness(video_u_planes: list, video_v_planes: list, bit_depth: int = 8) -> dict:
    """
    Utility function to process lists of U and V planes for a video segment.
    
    Args:
        video_u_planes (list): List of U planes (np.ndarray).
        video_v_planes (list): List of V planes (np.ndarray).
        bit_depth (int): The bit depth of the raw frames.
        
    Returns:
        dict: Statistical summary of the video's colorfulness.
    """
    if not video_u_planes or not video_v_planes or len(video_u_planes) != len(video_v_planes):
        return {"mean_colorfulness": 0.0, "std_colorfulness": 0.0, "max_colorfulness": 0.0, "min_colorfulness": 0.0}
        
    metrics = [
        calculate_hasler_suesstrunk_colorfulness_yuv(u, v, bit_depth) 
        for u, v in zip(video_u_planes, video_v_planes)
    ]
    
    return {
        
        "mean_colorfulness": float(np.mean(metrics)),
        "std_colorfulness": float(np.std(metrics)),
        "max_colorfulness": float(np.max(metrics)),
        "min_colorfulness": float(np.min(metrics))
    }
    
    