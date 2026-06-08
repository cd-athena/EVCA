import argparse
import os
import numpy as np

def generate_dummy_yuv(output_path: str, width: int = 128, height: int = 128, frames: int = 16):
    """
    Generates a synthetic, uncompressed YUV 4:2:0 8-bit video file.
    
    The video contains moving gradients to ensure that EVCA's Spatial Complexity (SC),
    Temporal Complexity (TC), and Colorfulness (C) metrics evaluate to non-zero values 
    during unit and integration testing.
    """
    # Ensure dimensions are even numbers (required for 4:2:0 chroma subsampling)
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("Width and height must be even numbers for 4:2:0 YUV.")

    # Calculate plane dimensions
    chroma_width = width // 2
    chroma_height = height // 2

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'wb') as f:
        for i in range(frames):
            # 1. Luma (Y) Plane: Moving diagonal gradient
            # Formula ensures high spatial edges (modulo wrapping) and temporal motion (i * 15)
            y_plane = np.fromfunction(
                lambda y, x: (x * 2 + y * 2 + i * 15) % 255,
                (height, width)
            ).astype(np.uint8)

            # 2. Chroma (U) Plane: Horizontal moving gradient (Blue-difference variance)
            u_plane = np.fromfunction(
                lambda y, x: (x * 4 + i * 10) % 255,
                (chroma_height, chroma_width)
            ).astype(np.uint8)

            # 3. Chroma (V) Plane: Vertical moving gradient (Red-difference variance)
            v_plane = np.fromfunction(
                lambda y, x: (y * 4 + i * 10) % 255,
                (chroma_height, chroma_width)
            ).astype(np.uint8)

            # Write raw bytes to file sequentially (Y, then U, then V)
            f.write(y_plane.tobytes())
            f.write(u_plane.tobytes())
            f.write(v_plane.tobytes())

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"Success: Generated synthetic YUV test asset.")
    print(f"Path: {output_path}")
    print(f"Specs: {width}x{height}, {frames} frames, 4:2:0 8-bit")
    print(f"Size: {file_size_kb:.2f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dummy YUV file for EVCA CI Testing.")
    parser.add_argument('-o', '--output', type=str, default='tests/dummy.yuv', help='Output file path')
    parser.add_argument('-W', '--width', type=int, default=128, help='Video width (must be even)')
    parser.add_argument('-H', '--height', type=int, default=128, help='Video height (must be even)')
    parser.add_argument('-f', '--frames', type=int, default=16, help='Number of frames to generate')
    
    args = parser.parse_args()
    
    generate_dummy_yuv(args.output, args.width, args.height, args.frames)