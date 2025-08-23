#!/usr/bin/env python3
"""
Complete Solar Differential Rotation Analysis Pipeline

This script integrates sunspot detection and tracking to measure
solar differential rotation from a time series of solar images.

Usage:
    python solar_pipeline.py --config config.json
    python solar_pipeline.py --images image1.fits image2.fits --times "2023-10-15T12:00" "2023-10-17T12:00"
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

# Import our custom classes
from sunspot_detection import SolarDiskDetector, SunspotDetector, main_analysis
from sunspot_tracking import SunspotTracker, DifferentialRotationAnalyzer
from solar_utilities import (SolarImageLoader, CoordinateTransformer, 
                            SolarDataDownloader, QualityAssessment, 
                            create_analysis_config_template)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarAnalysisPipeline:
    """
    Complete pipeline for solar differential rotation analysis
    """
    
    def __init__(self, config=None):
        """
        Initialize the analysis pipeline
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        
        # Default configuration
        self.config = {
            'detection': {
                'disk_method': 'hough',  # 'hough' or 'edge'
                'min_sunspot_area': 10,
                'intensity_threshold': None,
                'limb_darkening_correction': True
            },
            'tracking': {
                'max_distance_deg': 15.0,
                'max_time_days': 7.0,
                'min_track_length': 2
            },
            'analysis': {
                'fit_differential_rotation': True,
                'export_results': True,
                'create_plots': True
            },
            'output': {
                'results_dir': 'solar_analysis_results',
                'save_intermediate': True,
                'plot_format': 'png'
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(self.config, config)
        
        # Initialize components
        self.disk_detector = SolarDiskDetector()
        self.analyzer = DifferentialRotationAnalyzer()
        self.quality_assessor = QualityAssessment()
        
        # Storage for results
        self.processed_images = []
        self.analysis_results = None
        
        # Create output directory
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
    def _update_config(self, base_config, update_config):
        """Recursively update configuration dictionary"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def process_image_sequence(self, image_paths, observation_times):
        """
        Process a sequence of solar images
        
        Parameters:
        -----------
        image_paths : list
            List of paths to solar images
        observation_times : list  
            List of observation times (ISO format strings or datetime objects)
        
        Returns:
        --------
        success : bool
            True if processing successful
        """
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        if len(image_paths) != len(observation_times):
            raise ValueError("Number of images must match number of observation times")
        
        self.processed_images = []
        
        for i, (image_path, obs_time) in enumerate(zip(image_paths, observation_times)):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Process individual image using updated main_analysis function
                sunspots, disk_info = main_analysis(
                    image_path, 
                    obs_time,
                    min_area=self.config['detection']['min_sunspot_area'],
                    intensity_threshold=self.config['detection']['intensity_threshold'],
                    disk_method=self.config['detection']['disk_method']
                )
                
                # Store results
                processed_data = {
                    'image_path': image_path,
                    'observation_time': obs_time,
                    'sunspots': sunspots,
                    'disk_info': disk_info,
                    'processing_successful': True
                }
                
                self.processed_images.append(processed_data)
                
                # Save intermediate results if requested
                if self.config['output']['save_intermediate']:
                    self._save_intermediate_results(processed_data, i)
                
                logger.info(f"  Found {len(sunspots)} sunspots")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                # Add failed entry to maintain sequence
                self.processed_images.append({
                    'image_path': image_path,
                    'observation_time': obs_time,
                    'sunspots': [],
                    'disk_info': None,
                    'processing_successful': False,
                    'error': str(e)
                })
        
        # Filter out failed processing
        successful_images = [img for img in self.processed_images if img['processing_successful']]
        
        logger.info(f"Successfully processed {len(successful_images)}/{len(image_paths)} images")
        
        return len(successful_images) > 1  # Need at least 2 for tracking
    
    def run_tracking_analysis(self):
        """
        Run sunspot tracking and differential rotation analysis
        
        Returns:
        --------
        results : dict
            Analysis results
        """
        
        logger.info("Starting tracking analysis...")
        
        # Filter successful images
        successful_images = [img for img in self.processed_images if img['processing_successful']]
        
        if len(successful_images) < 2:
            logger.error("Need at least 2 successfully processed images for tracking")
            return None
        
        # Prepare data for tracking
        image_data_list = []
        observation_times = []
        
        for img_data in successful_images:
            image_data_list.append((
                img_data['image_path'],
                img_data['sunspots'],
                img_data['disk_info']
            ))
            observation_times.append(img_data['observation_time'])
        
        # Run analysis
        self.analysis_results = self.analyzer.analyze_image_sequence(
            image_data_list, observation_times
        )
        
        if self.analysis_results is None:
            logger.error("Tracking analysis failed")
            return None
        
        # Compare with literature if fit was successful
        if self.analysis_results['fit_parameters']:
            logger.info("Comparing with literature values...")
            self.analyzer.compare_with_literature(self.analysis_results['fit_parameters'])
        
        # Save results
        if self.config['analysis']['export_results']:
            self._save_final_results()
        
        logger.info("Analysis complete!")
        return self.analysis_results
    
    def _save_intermediate_results(self, processed_data, index):
        """Save intermediate processing results"""
        
        output_file = self.results_dir / f"image_{index:03d}_results.json"
        
        # Convert data for JSON serialization
        json_data = {
            'image_path': str(processed_data['image_path']),
            'observation_time': str(processed_data['observation_time']),
            'num_sunspots': len(processed_data['sunspots']),
            'processing_successful': processed_data['processing_successful']
        }
        
        if processed_data['processing_successful']:
            json_data['disk_center'] = processed_data['disk_info']['center']
            json_data['disk_radius'] = processed_data['disk_info']['radius']
            
            # Sunspot summary
            sunspot_summary = []
            for i, spot in enumerate(processed_data['sunspots']):
                spot_data = {
                    'id': i,
                    'area': float(spot['area']),
                    'centroid': [float(spot['centroid'][0]), float(spot['centroid'][1])],
                    'mean_intensity': float(spot['mean_intensity'])
                }
                
                # Add coordinates if available
                if 'heliographic_lon' in spot:
                    spot_data['heliographic_lon'] = float(spot['heliographic_lon'].to_value('deg'))
                    spot_data['heliographic_lat'] = float(spot['heliographic_lat'].to_value('deg'))
                
                sunspot_summary.append(spot_data)
            
            json_data['sunspots'] = sunspot_summary
        else:
            json_data['error'] = processed_data.get('error', 'Unknown error')
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _save_final_results(self):
        """Save final analysis results"""
        
        if self.analysis_results is None:
            return
        
        # Save CSV data
        csv_file = self.results_dir / 'rotation_analysis.csv'
        if 'results_dataframe' in self.analysis_results:
            self.analysis_results['results_dataframe'].to_csv(csv_file, index=False)
            logger.info(f"Saved rotation data to {csv_file}")
        
        # Save fit parameters
        if self.analysis_results['fit_parameters']:
            fit_file = self.results_dir / 'differential_rotation_fit.json'
            
            fit_data = self.analysis_results['fit_parameters'].copy()
            # Convert any numpy types for JSON serialization
            for key, value in fit_data.items():
                if isinstance(value, np.floating):
                    fit_data[key] = float(value)
            
            with open(fit_file, 'w') as f:
                json.dump(fit_data, f, indent=2)
            
            logger.info(f"Saved fit parameters to {fit_file}")
        
        # Save track summary
        track_file = self.results_dir / 'sunspot_tracks.json'
        track_summary = []
        
        for track in self.analysis_results['tracks']:
            track_data = {
                'track_id': track['track_id'],
                'num_observations': len(track['observations']),
                'active': track['active']
            }
            track_summary.append(track_data)
        
        with open(track_file, 'w') as f:
            json.dump(track_summary, f, indent=2)
        
        logger.info(f"Saved track summary to {track_file}")
    
    def create_summary_report(self):
        """Create a summary report of the analysis"""
        
        if not self.analysis_results:
            logger.warning("No analysis results to report")
            return
        
        report_file = self.results_dir / 'analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("Solar Differential Rotation Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Processing summary
            total_images = len(self.processed_images)
            successful_images = len([img for img in self.processed_images if img['processing_successful']])
            f.write(f"Images processed: {successful_images}/{total_images}\n")
            
            # Sunspot detection summary
            total_sunspots = sum(len(img['sunspots']) for img in self.processed_images if img['processing_successful'])
            f.write(f"Total sunspots detected: {total_sunspots}\n")
            
            # Tracking results
            if 'tracks' in self.analysis_results:
                num_tracks = len(self.analysis_results['tracks'])
                f.write(f"Sunspot tracks created: {num_tracks}\n")
            
            # Rotation analysis
            if 'rotation_data' in self.analysis_results:
                num_measured = len(self.analysis_results['rotation_data'])
                f.write(f"Rotation rates measured: {num_measured}\n")
            
            # Fit results
            if self.analysis_results['fit_parameters']:
                fit = self.analysis_results['fit_parameters']
                f.write(f"\nDifferential Rotation Fit:\n")
                f.write(f"  A = {fit['A']:.3f} ± {fit['A_error']:.3f} deg/day\n")
                f.write(f"  B = {fit['B']:.3f} ± {fit['B_error']:.3f} deg/day\n") 
                f.write(f"  C = {fit['C']:.3f} ± {fit['C_error']:.3f} deg/day\n")
                f.write(f"  R-squared = {fit['r_squared']:.4f}\n")
                f.write(f"  Equatorial period = {fit['equatorial_period']:.2f} days\n")
            
            f.write(f"\nAnalysis completed: {datetime.now().isoformat()}\n")
        
        logger.info(f"Summary report saved to {report_file}")


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(
        description="Solar Differential Rotation Analysis Pipeline"
    )
    
    parser.add_argument('--config', type=str, help='Configuration file (JSON)')
    parser.add_argument('--images', nargs='+', help='List of solar image files')
    parser.add_argument('--times', nargs='+', help='Observation times (ISO format)')
    parser.add_argument('--output-dir', type=str, default='solar_analysis_results',
                       help='Output directory')
    parser.add_argument('--create-config', action='store_true',
                       help='Create example configuration file and exit')
    
    args = parser.parse_args()
    
    # Create config template if requested
    if args.create_config:
        config_template = create_analysis_config_template()
        with open('solar_analysis_config.json', 'w') as f:
            json.dump(config_template, f, indent=2)
        print("Configuration template saved to 'solar_analysis_config.json'")
        print("Edit this file and run with --config solar_analysis_config.json")
        return
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return
    
    if args.output_dir:
        config['output'] = config.get('output', {})
        config['output']['results_dir'] = args.output_dir
    
    # Initialize pipeline
    pipeline = SolarAnalysisPipeline(config)
    
    # Get image list and times
    if args.images and args.times:
        image_paths = args.images
        observation_times = args.times
    else:
        logger.error("Must provide --images and --times arguments")
        logger.info("Example usage:")
        logger.info("  python solar_pipeline.py --images img1.fits img2.fits --times '2023-10-15T12:00' '2023-10-17T12:00'")
        return
    
    if len(image_paths) != len(observation_times):
        logger.error("Number of images must match number of times")
        return
    
    try:
        # Process images
        success = pipeline.process_image_sequence(image_paths, observation_times)
        
        if not success:
            logger.error("Image processing failed")
            return
        
        # Run tracking analysis
        results = pipeline.run_tracking_analysis()
        
        if results:
            # Create summary report
            pipeline.create_summary_report()
            logger.info("Analysis completed successfully!")
        else:
            logger.error("Tracking analysis failed")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Example usage function
def example_workflow():
    """
    Example workflow for complete sunspot tracking analysis
    """
    
    print("Example Workflow for Solar Differential Rotation Analysis")
    print("=" * 60)
    
    # Create example configuration
    config = create_analysis_config_template()
    
    # Example image paths and times
    image_paths = [
        "solar_image_day1.fits",
        "solar_image_day3.fits", 
        "solar_image_day5.fits",
        "solar_image_day7.fits"
    ]
    
    observation_times = [
        "2023-10-15T12:00:00",
        "2023-10-17T12:00:00",
        "2023-10-19T12:00:00",
        "2023-10-21T12:00:00"
    ]
    
    print("Step 1: Initialize pipeline with configuration")
    pipeline = SolarAnalysisPipeline(config)
    
    print("Step 2: Process image sequence")
    # success = pipeline.process_image_sequence(image_paths, observation_times)
    
    print("Step 3: Run tracking analysis")
    # results = pipeline.run_tracking_analysis()
    
    print("Step 4: Create summary report")
    # pipeline.create_summary_report()
    
    print("\nRequired packages:")
    print("pip install numpy matplotlib astropy sunpy opencv-python scikit-image scipy pandas")
    
    print("\nCommand line usage:")
    print("python solar_pipeline.py --create-config  # Create configuration template")
    print("python solar_pipeline.py --config config.json")
    print("python solar_pipeline.py --images img1.fits img2.fits --times '2023-10-15T12:00' '2023-10-17T12:00'")


# Run example if called directly
if __name__ == "__main__" and len(__import__('sys').argv) == 1:
    example_workflow()