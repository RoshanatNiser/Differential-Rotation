#!/usr/bin/env python3
"""
Sample Analysis Scripts for Solar Differential Rotation Pipeline

This file contains example usage patterns for the solar rotation analysis pipeline.
"""

import sys
import os
from pathlib import Path
import json

# Add parent directory to path to import pipeline modules
sys.path.append(str(Path(__file__).parent.parent))

from solar_pipeline import SolarAnalysisPipeline
from solar_utilities import create_analysis_config_template

def example_1_basic_usage():
    """
    Example 1: Basic usage with default configuration
    """
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Define image files and observation times
    image_files = [
        "sample_images/solar_001.fits",
        "sample_images/solar_002.fits", 
        "sample_images/solar_003.fits"
    ]
    
    observation_times = [
        "2023-10-15T12:00:00",
        "2023-10-17T12:00:00",
        "2023-10-19T12:00:00"
    ]
    
    # Initialize pipeline with default configuration
    pipeline = SolarAnalysisPipeline()
    
    # Process images
    print("Processing image sequence...")
    success = pipeline.process_image_sequence(image_files, observation_times)
    
    if success:
        print("Images processed successfully!")
        
        # Run tracking analysis
        print("Running tracking analysis...")
        results = pipeline.run_tracking_analysis()
        
        if results:
            print(f"Analysis complete! Found {len(results['tracks'])} sunspot tracks")
            
            # Create summary report
            pipeline.create_summary_report()
            
            return results
        else:
            print("Tracking analysis failed")
    else:
        print("Image processing failed")
    
    return None

def example_2_custom_configuration():
    """
    Example 2: Using custom configuration
    """
    print("="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    config = {
        'detection': {
            'min_sunspot_area': 5,      # More sensitive detection
            'disk_method': 'hough',
            'intensity_threshold': None
        },
        'tracking': {
            'max_distance_deg': 20.0,   # More permissive tracking
            'max_time_days': 7.0,
            'min_track_length': 2
        },
        'output': {
            'results_dir': 'custom_results',
            'save_intermediate': True,
            'plot_format': 'png'
        }
    }
    
    # Initialize pipeline with custom config
    pipeline = SolarAnalysisPipeline(config)
    
    # Sample data
    image_files = [
        "sample_images/hmi_001.fits",
        "sample_images/hmi_002.fits",
        "sample_images/hmi_003.fits",
        "sample_images/hmi_004.fits"
    ]
    
    observation_times = [
        "2023-08-10T12:00:00",
        "2023-08-12T12:00:00", 
        "2023-08-14T12:00:00",
        "2023-08-16T12:00:00"
    ]
    
    # Process
    success = pipeline.process_image_sequence(image_files, observation_times)
    
    if success:
        results = pipeline.run_tracking_analysis()
        
        if results and results['fit_parameters']:
            fit = results['fit_parameters']
            print(f"\\nDifferential Rotation Results:")
            print(f"A = {fit['A']:.3f} ± {fit['A_error']:.3f} deg/day")
            print(f"B = {fit['B']:.3f} ± {fit['B_error']:.3f} deg/day")
            print(f"C = {fit['C']:.3f} ± {fit['C_error']:.3f} deg/day")
            print(f"R² = {fit['r_squared']:.4f}")
            print(f"Equatorial period = {fit['equatorial_period']:.2f} days")
        
        return results
    
    return None

def example_3_batch_processing():
    """
    Example 3: Batch processing multiple datasets
    """
    print("="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Define multiple datasets
    datasets = [
        {
            'name': 'Dataset_2023_Oct',
            'images': [
                'data/oct2023/solar_001.fits',
                'data/oct2023/solar_002.fits',
                'data/oct2023/solar_003.fits'
            ],
            'times': [
                '2023-10-15T12:00:00',
                '2023-10-17T12:00:00',
                '2023-10-19T12:00:00'
            ],
            'config_overrides': {
                'detection': {'min_sunspot_area': 10},
                'output': {'results_dir': 'results_oct2023'}
            }
        },
        {
            'name': 'Dataset_2023_Nov',
            'images': [
                'data/nov2023/solar_001.fits',
                'data/nov2023/solar_002.fits',
                'data/nov2023/solar_003.fits'
            ],
            'times': [
                '2023-11-05T12:00:00',
                '2023-11-07T12:00:00',
                '2023-11-09T12:00:00'
            ],
            'config_overrides': {
                'detection': {'min_sunspot_area': 15},
                'output': {'results_dir': 'results_nov2023'}
            }
        }
    ]
    
    results_summary = []
    
    for dataset in datasets:
        print(f"\\nProcessing {dataset['name']}...")
        
        # Create base configuration
        config = create_analysis_config_template()
        
        # Apply dataset-specific overrides
        if 'config_overrides' in dataset:
            for category, overrides in dataset['config_overrides'].items():
                if category in config:
                    config[category].update(overrides)
        
        # Initialize pipeline
        pipeline = SolarAnalysisPipeline(config)
        
        # Process dataset
        success = pipeline.process_image_sequence(
            dataset['images'], 
            dataset['times']
        )
        
        if success:
            results = pipeline.run_tracking_analysis()
            
            if results:
                # Store summary
                summary = {
                    'dataset_name': dataset['name'],
                    'num_images': len(dataset['images']),
                    'num_tracks': len(results['tracks']),
                    'fit_quality': results['fit_parameters']['r_squared'] if results['fit_parameters'] else None
                }
                results_summary.append(summary)
                
                pipeline.create_summary_report()
                print(f"  ✓ Success: {summary['num_tracks']} tracks found")
            else:
                print(f"  ✗ Failed: Analysis unsuccessful")
        else:
            print(f"  ✗ Failed: Image processing unsuccessful")
    
    # Print batch summary
    print(f"\\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    for summary in results_summary:
        print(f"{summary['dataset_name']}: "
              f"{summary['num_tracks']} tracks, "
              f"R² = {summary['fit_quality']:.3f}" if summary['fit_quality'] else "No fit")
    
    return results_summary

def example_4_advanced_analysis():
    """
    Example 4: Advanced analysis with result interpretation
    """
    print("="*60)
    print("EXAMPLE 4: Advanced Analysis")
    print("="*60)
    
    # Load configuration from file
    config_file = Path(__file__).parent / "sample_config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_file}")
    else:
        config = create_analysis_config_template()
        print("Using default configuration template")
    
    # High-quality analysis settings
    config['detection']['min_sunspot_area'] = 8
    config['tracking']['min_track_length'] = 3
    config['analysis']['minimum_tracks_for_fit'] = 5
    
    # Initialize pipeline
    pipeline = SolarAnalysisPipeline(config)
    
    # Sample high-quality dataset
    image_files = [
        "hq_data/hmi_20231015_120000.fits",
        "hq_data/hmi_20231016_120000.fits",
        "hq_data/hmi_20231017_120000.fits",
        "hq_data/hmi_20231018_120000.fits",
        "hq_data/hmi_20231019_120000.fits"
    ]
    
    observation_times = [
        "2023-10-15T12:00:00",
        "2023-10-16T12:00:00",
        "2023-10-17T12:00:00",
        "2023-10-18T12:00:00",
        "2023-10-19T12:00:00"
    ]
    
    # Process
    success = pipeline.process_image_sequence(image_files, observation_times)
    
    if success:
        results = pipeline.run_tracking_analysis()
        
        if results:
            # Detailed analysis
            analyze_results_quality(results)
            compare_with_literature(results)
            
            return results
    
    return None

def analyze_results_quality(results):
    """Analyze and report on result quality"""
    
    print("\\n" + "="*50)
    print("RESULTS QUALITY ANALYSIS")
    print("="*50)
    
    tracks = results['tracks']
    rotation_data = results['rotation_data']
    fit_params = results['fit_parameters']
    
    # Track quality metrics
    track_lengths = [len(track['observations']) for track in tracks]
    active_tracks = [track for track in tracks if track['active']]
    
    print(f"Total tracks created: {len(tracks)}")
    print(f"Active tracks: {len(active_tracks)}")
    print(f"Average track length: {np.mean(track_lengths):.1f} observations")
    print(f"Tracks used for rotation fit: {len(rotation_data)}")
    
    if fit_params:
        print(f"\\nFit Quality:")
        print(f"R-squared: {fit_params['r_squared']:.4f}")
        
        if fit_params['r_squared'] > 0.9:
            print("  → Excellent fit quality")
        elif fit_params['r_squared'] > 0.7:
            print("  → Good fit quality") 
        elif fit_params['r_squared'] > 0.5:
            print("  → Moderate fit quality")
        else:
            print("  → Poor fit quality - consider more data")
    
    # Latitude coverage
    if rotation_data:
        latitudes = [r['mean_latitude'] for r in rotation_data]
        lat_range = max(latitudes) - min(latitudes)
        print(f"\\nLatitude coverage: {min(latitudes):.1f}° to {max(latitudes):.1f}° ({lat_range:.1f}° range)")
        
        if lat_range > 40:
            print("  → Excellent latitude coverage")
        elif lat_range > 20:
            print("  → Good latitude coverage")
        else:
            print("  → Limited latitude coverage - results may be less reliable")

def compare_with_literature(results):
    """Compare results with literature values"""
    
    if not results['fit_parameters']:
        return
    
    fit = results['fit_parameters']
    
    print("\\n" + "="*50)
    print("LITERATURE COMPARISON")
    print("="*50)
    
    # Snodgrass & Ulrich (1990) values
    lit_A, lit_B, lit_C = 14.713, -2.396, -1.787
    
    print(f"Parameter A (equatorial rate):")
    print(f"  Your result: {fit['A']:.3f} ± {fit['A_error']:.3f} deg/day")
    print(f"  Literature:  {lit_A:.3f} deg/day")
    print(f"  Difference:  {abs(fit['A'] - lit_A):.3f} deg/day")
    
    print(f"\\nParameter B:")
    print(f"  Your result: {fit['B']:.3f} ± {fit['B_error']:.3f} deg/day")
    print(f"  Literature:  {lit_B:.3f} deg/day")
    print(f"  Difference:  {abs(fit['B'] - lit_B):.3f} deg/day")
    
    print(f"\\nParameter C:")
    print(f"  Your result: {fit['C']:.3f} ± {fit['C_error']:.3f} deg/day")
    print(f"  Literature:  {lit_C:.3f} deg/day")
    print(f"  Difference:  {abs(fit['C'] - lit_C):.3f} deg/day")
    
    # Agreement assessment
    a_agree = abs(fit['A'] - lit_A) < 2 * fit['A_error']
    b_agree = abs(fit['B'] - lit_B) < 2 * fit['B_error'] 
    c_agree = abs(fit['C'] - lit_C) < 2 * fit['C_error']
    
    agreement_score = sum([a_agree, b_agree, c_agree])
    print(f"\\nAgreement with literature: {agreement_score}/3 parameters within 2σ")
    
    if agreement_score == 3:
        print("  → Excellent agreement with literature!")
    elif agreement_score == 2:
        print("  → Good agreement with literature")
    else:
        print("  → Limited agreement - check data quality and methods")

def main():
    """Run all examples"""
    
    print("Solar Differential Rotation Analysis - Example Usage")
    print("="*60)
    
    # Note: These examples assume you have sample data files
    # Uncomment and modify paths as needed for your data
    
    print("\\nNOTE: Examples require sample data files.")
    print("Modify image file paths in the examples to match your data.")
    print("\\nAvailable examples:")
    print("1. Basic usage with default settings")
    print("2. Custom configuration")  
    print("3. Batch processing multiple datasets")
    print("4. Advanced analysis with quality assessment")
    
    # Uncomment to run examples:
    # results1 = example_1_basic_usage()
    # results2 = example_2_custom_configuration()
    # summary = example_3_batch_processing()
    # results4 = example_4_advanced_analysis()

if __name__ == "__main__":
    main()
