import unittest
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

# Import our dummy generator from Phase 1
from tests.generate_test_yuv import generate_dummy_yuv

class TestEVCA_CLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Runs once before any tests start.
        Generates the 128x128 16-frame dummy video required for CLI testing.
        """
        cls.dummy_path = 'tests/dummy_integration.yuv'
        cls.frames = 16
        generate_dummy_yuv(cls.dummy_path, width=128, height=128, frames=cls.frames)

    @classmethod
    def tearDownClass(cls):
        """
        Runs once after all tests finish. Cleans up the dummy video.
        (We leave the CSV/PNG artifacts so developers can inspect them if a test fails).
        """
        if os.path.exists(cls.dummy_path):
            os.remove(cls.dummy_path)

    def run_cli_command(self, args_list):
        """
        Helper method to execute main.py with arguments and assert a clean exit.
        Uses sys.executable to ensure it runs in the current virtual environment.
        """
        cmd = [sys.executable, 'main.py', '-i', self.dummy_path, '-r', '128x128'] + args_list
        
        # Run command, capturing standard out and error
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # If the command crashes, print the stack trace for debugging
        self.assertEqual(
            result.returncode, 0, 
            f"CLI Command Failed! Arguments: {args_list}\nError Log:\n{result.stderr}"
        )
        return result

    def test_01_standard_evca(self):
        """Test default EVCA execution and CSV artifact generation."""
        csv_out = './csv/test_evca_default.csv'
        self.run_cli_command(['-c', csv_out])
        
        self.assertTrue(os.path.exists(csv_out), "CSV file was not created.")
        
        df = pd.read_csv(csv_out)
        self.assertEqual(len(df), self.frames, "CSV row count does not match frame count.")
        self.assertIn('TC', df.columns, "EVCA specific column 'TC' missing.")
        self.assertNotIn('Colorfulness', df.columns, "Colorfulness should not exist by default.")

    def test_02_vca_mode(self):
        """Test the legacy VCA method."""
        csv_out = './csv/test_vca_mode.csv'
        self.run_cli_command(['-m', 'VCA', '-c', csv_out])
        
        df = pd.read_csv(csv_out)
        self.assertIn('h', df.columns, "VCA specific column 'h' missing.")

    def test_03_siti_mode(self):
        """Test the SITI method."""
        csv_out = './csv/test_siti_mode.csv'
        self.run_cli_command(['-m', 'SITI', '-c', csv_out])
        
        df = pd.read_csv(csv_out)
        self.assertIn('TI', df.columns, "SITI execution failed to write correct headers.")

    def test_04_colorfulness_flag(self):
        """Test that the --colorfulness flag injects the new metric correctly."""
        csv_out = './csv/test_color_flag.csv'
        self.run_cli_command(['-cf', '-c', csv_out])
        
        df = pd.read_csv(csv_out)
        self.assertIn('Colorfulness', df.columns, "Colorfulness flag did not append to CSV.")
        
        # Assert the metric is mathematically viable (not just all zeros or NaNs)
        self.assertTrue(df['Colorfulness'].mean() > 0.0, "Colorfulness metric computed as <= 0 on colorful dummy video.")

    def test_05_subsampling_regression(self):
        """
        CRITICAL TEST: Ensures the tensor double-subsampling bug does not return.
        Tests sample_rate = 4.
        """
        csv_out = './csv/test_subsampling.csv'
        self.run_cli_command(['-s', '4', '--colorfulness', '-c', csv_out])
        
        df = pd.read_csv(csv_out)
        expected_rows = self.frames // 4
        self.assertEqual(
            len(df), expected_rows, 
            f"Subsampling failed. Expected {expected_rows} rows, got {len(df)}."
        )

    def test_06_plotting_flags(self):
        """Test the -pi and -pm visual plotting flags and artifact routing."""
        csv_out = './csv/test_plots.csv'
        self.run_cli_command(['-pi', '1', '-pm', '1', '--colorfulness', '-c', csv_out])
        
        # Check standard EVCA block-metric plot
        block_plot = './png/EVCA_frame_000.png'
        self.assertTrue(os.path.exists(block_plot), "Block plot was not generated.")
        
        # Check our new Frame Metrics plot in its dedicated directory
        frame_plot = './png/frame_metrics/test_plots_EVCA_dummy_integration_frame_metrics.png'
        self.assertTrue(os.path.exists(frame_plot), "Frame metrics plot was not generated in the correct target directory.")
        
    def test_07_chroma_complexity_flag(self):
        """Test the --chroma_complexity flat to enable calculation of the Spatial Chroma Complexity (SC_c)."""
        csv_out = './csv/test_scc.csv'
        self.run_cli_command(['--chroma_complexity', '-c', csv_out, '-s', '2',])
        
        df = pd.read_csv(csv_out)
        
        # Verify the U channel spatial complexity
        self.assertIn(
            'SC_u', df.columns,
            f"Chroma complexity column 'SC_u' is missing from CSV headers: {list(df.columns)}."
        )
        
        # Verify the V channel spatial complexity
        self.assertIn(
            'SC_v', df.columns,
            f"Chroma complexity column 'SC_v' is missing from CSV headers: {list(df.columns)}."
        )

if __name__ == '__main__':
    unittest.main()