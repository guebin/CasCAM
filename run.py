#!/usr/bin/env python3
"""
CasCAM Analysis Runner

This script runs the complete CasCAM (Cascaded Class Activation Mapping) analysis pipeline.
Based on the original 2025-08-19-pets-cscam.py implementation.

Usage:
    python run.py [--config CONFIG_FILE]
    
Example:
    python run.py  # Use default configuration
    python run.py --config custom_config.py
"""

import argparse
import sys
from core import CasCAMConfig, CasCAMAnalyzer


def create_default_config():
    """Create default CasCAM configuration"""
    return CasCAMConfig(
        num_iter=3,
        theta=0.1,
        lambda_vals=[0.369],
        data_path="./data/pet/",
        random_seed=43052,
        max_comparison_images=None
    )


def load_custom_config(config_file):
    """Load configuration from custom file"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        print("Using default configuration instead.")
        return create_default_config()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run CasCAM Analysis')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--num_iter', type=int, help='Override number of iterations')
    parser.add_argument('--theta', type=float, help='Override theta parameter')
    parser.add_argument('--lambda_vals', type=float, nargs='+', help='Override lambda parameters (single or multiple values)')
    parser.add_argument('--data_path', type=str, help='Override original data path')
    parser.add_argument('--max_comparison_images', type=int, help='Maximum number of images for comparison (default: all)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_custom_config(args.config)
    else:
        config = create_default_config()
    
    # Handle lambda values
    lambda_values = args.lambda_vals if args.lambda_vals is not None else config.lambda_vals
    
    # Create single config with all lambda values
    final_config = CasCAMConfig(
        num_iter=args.num_iter if args.num_iter is not None else config.num_iter,
        theta=args.theta if args.theta is not None else config.theta,
        lambda_vals=lambda_values,
        data_path=args.data_path if args.data_path is not None else config.data_path,
        random_seed=config.random_seed,
        max_comparison_images=args.max_comparison_images if args.max_comparison_images is not None else config.max_comparison_images
    )
    
    print(f"Running CasCAM: {final_config.dataset_name} | θ={final_config.theta} | λ={lambda_values} | {final_config.num_iter} iter")
    
    try:
        # Run analysis once for all lambda values
        analyzer = CasCAMAnalyzer(final_config)
        all_results = analyzer.run_full_analysis()
        
        print(f"✓ Analysis complete: {len(lambda_values)} lambda values processed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        all_results = {}
    
    # Final summary
    if len(all_results) > 1:
        print(f"\nBatch complete: {len(all_results)} lambda values processed")


if __name__ == "__main__":
    main()