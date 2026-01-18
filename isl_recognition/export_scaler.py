#!/usr/bin/env python3
"""
Export scaler parameters to JSON for Flutter app.

Usage:
    python export_scaler.py --input data/processed --output ../isl_app/assets/
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np


def export_scaler(input_dir: str, output_dir: str):
    """Export sklearn StandardScaler to JSON format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load scaler
    scaler = joblib.load(input_path / 'scaler.pkl')
    
    # Extract parameters
    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()  # sklearn stores scale_ = std
    }
    
    # Save to JSON
    output_file = output_path / 'scaler.json'
    with open(output_file, 'w') as f:
        json.dump(scaler_data, f)
    
    print(f"Exported scaler to: {output_file}")
    print(f"  Features: {len(scaler_data['mean'])}")
    

def main():
    parser = argparse.ArgumentParser(description='Export scaler to JSON')
    parser.add_argument('--input', type=str, default='data/processed',
                       help='Path to processed data containing scaler.pkl')
    parser.add_argument('--output', type=str, default='../isl_app/assets',
                       help='Output directory for scaler.json')
    
    args = parser.parse_args()
    export_scaler(args.input, args.output)


if __name__ == '__main__':
    main()
