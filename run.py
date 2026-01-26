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
from config import CasCAMConfig
from analyzer import CasCAMAnalyzer


def create_default_config():
    """Create default CasCAM configuration"""
    return CasCAMConfig(
        num_iter=3,
        theta=0.3,
        lambda_vals=[0.1],
        data_path="./data/oxford-pets-cascam/with_artifact/",
        random_seed=43052,
        max_comparison_images=None,
        threshold_method='top_k',
        threshold_params={'k': 10}
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
    parser.add_argument('--num_iter', type=int, default=3, help='Number of iterations (default: 3)')
    parser.add_argument('--theta', type=float, default=0.3, help='Theta parameter (default: 0.3)')
    parser.add_argument('--lambda_vals', type=float, nargs='+', default=[0.1], help='Lambda values (default: [0.1])')
    parser.add_argument('--data_path', type=str, default="./data/oxford-pets-cascam/with_artifact/", help='Data path (default: ./data/oxford-pets-cascam/with_artifact/)')
    parser.add_argument('--random_seed', type=int, default=43052, help='Random seed (default: 43052)')
    parser.add_argument('--max_comparison_images', type=int, default=None, help='Maximum number of images for comparison (default: all)')
    parser.add_argument('--threshold_method', type=str, choices=['top_k', 'ebayesthresh'], default=None, help='CAM thresholding method (omit for no thresholding)')
    parser.add_argument('--top_k', type=float, default=10, help='Top-k percentage for top_k method (default: 10)')
    parser.add_argument('--ebayesthresh_method', type=str, choices=['sure', 'bayes'], default='sure', help='EBayesThresh method (default: sure)')
    parser.add_argument('--ebayesthresh_prior', type=str, choices=['laplace', 'cauchy'], default='laplace', help='EBayesThresh prior (default: laplace)')
    parser.add_argument('--ebayesthresh_a', type=float, default=0.5, help='EBayesThresh a parameter (default: 0.5)')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum training epochs (default: 10)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')

    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_custom_config(args.config)
    else:
        config = create_default_config()
    
    # Handle lambda values
    lambda_values = args.lambda_vals if args.lambda_vals is not None else config.lambda_vals
    
    # Setup threshold parameters
    threshold_params = {}
    if args.threshold_method == 'top_k':
        threshold_params = {'k': args.top_k}
    elif args.threshold_method == 'ebayesthresh':
        threshold_params = {
            'method': args.ebayesthresh_method,
            'prior': args.ebayesthresh_prior,
            'a': args.ebayesthresh_a
        }
    
    # Create single config with all lambda values
    final_config = CasCAMConfig(
        num_iter=args.num_iter,
        theta=args.theta,
        lambda_vals=lambda_values,
        data_path=args.data_path,
        random_seed=args.random_seed,
        max_comparison_images=args.max_comparison_images,
        threshold_method=args.threshold_method,
        threshold_params=threshold_params,
        max_epochs=args.max_epochs,
        patience=args.patience
    )
    
    print(f"Running CasCAM: {final_config.dataset_name} | θ={final_config.theta} | λ={lambda_values} | {final_config.num_iter} iter")
    
    try:
        # Run analysis once for all lambda values
        analyzer = CasCAMAnalyzer(final_config)
        all_results = analyzer.run_full_analysis()

        print(f"✓ Analysis complete: {len(lambda_values)} lambda values processed")

        # Save computation times
        analyzer.save_computation_times()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        all_results = {}

    # Final summary
    if len(all_results) > 1:
        print(f"\nBatch complete: {len(all_results)} lambda values processed")


if __name__ == "__main__":
    main()