#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar Analysis Utilities

Helper functions for solar image processing, coordinate transformations,
and data handling for sunspot analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
from astropy.wcs import WCS
from scipy import ndimage
import cv2
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# SunPy imports with fallback handling
try:
    import sunpy
    import sunpy.map
    from sunpy.coordinates import frames, get_earth
    SUNPY_AVAILABLE = True
except ImportError:
    SUNPY_AVAILABLE = False
    print("Warning: SunPy not available. Some coordinate transformations may be limited.")

# SunPy.net imports with fallback handling  
try:
    from sunpy.net import Fido, attrs as a
    from sunpy.time import parse_time
    SUNPY_NET_AVAILABLE = True
except ImportError:
    SUNPY_NET_AVAILABLE = False
    print("Warning: SunPy.net not available. Data downloading functionality disabled.")


class SolarImageLoader:
    """
    Utility class to load and standardize solar images from different sources
    """
    
    @staticmethod
    def load_fits_with_header(filepath):
        """
        Load FITS file with proper header handling
        
        Parameters:
        -----------
        filepath : str or Path
            Path to FITS file
            
        Returns:
        --------
        image_data : numpy.ndarray
            Image data array
        header : astropy.io.fits.Header
            FITS header
        metadata : dict
            Extracted metadata
        """
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")
        
        with fits.open(filepath) as hdul:
            # Get primary HDU or first extension with data
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None:
                    data_hdu = hdu
                    break
            
            if data_hdu is None:
                raise ValueError(f"No data found in FITS file: {filepath}")
            
            image_data = data_hdu.data
            header = data_hdu.header
            
            # Handle different data orientations
            if len(image_data.shape) > 2:
                # Take the first 2D slice if it's a cube
                image_data = image_data[0] if image_data.shape[0] < image_data.shape[-1] else image_data[:,:,0]
            
            # Extract common metadata
            metadata = SolarImageLoader._extract_metadata(header)
            
        return image_data, header, metadata
    
    @staticmethod
    def _extract_metadata(header):
        """Extract common solar observation metadata from FITS header"""
        
        metadata = {}
        
        # Common observation parameters
        obs_keys = [
            ('DATE-OBS', 'observation_date'),
            ('TIME-OBS', 'observation_time'),
            ('DATE_OBS', 'observation_date'),  # Alternative format
            ('OBSRVTRY', 'observatory'),
            ('INSTRUME', 'instrument'),
            ('TELESCOP', 'telescope'),
            ('FILTER', 'filter'),
            ('WAVELNTH', 'wavelength'),
            ('WAVEUNIT', 'wavelength_unit'),
            ('EXPTIME', 'exposure_time'),
            ('CDELT1', 'pixel_scale_x'),
            ('CDELT2', 'pixel_scale_y'),
            ('CRPIX1', 'reference_pixel_x'),
            ('CRPIX2', 'reference_pixel_y'),
            ('CRVAL1', 'reference_coord_x'),
            ('CRVAL2', 'reference_coord_y'),
            ('CTYPE1', 'coord_type_x'),
            ('CTYPE2', 'coord_type_y'),
            ('CUNIT1', 'coord_unit_x'),
            ('CUNIT2', 'coord_unit_y'),
            ('RSUN_REF', 'solar_radius_reference'),
            ('DSUN_OBS', 'sun_observer_distance'),
            ('SOLAR_R', 'solar_radius_pixels'),
            ('SOLAR_P0', 'solar_p_angle'),
            ('SOLAR_B0', 'solar_b_angle'),
            ('SOLAR_L0', 'solar_l_angle')
        ]
        
        for fits_key, meta_key in obs_keys:
            if fits_key in header:
                metadata[meta_key] = header[fits_key]
        
        # Combine date and time if separate
        if 'observation_date' in metadata and 'observation_time' in metadata:
            try:
                date_str = str(metadata['observation_date'])
                time_str = str(metadata['observation_time'])
                metadata['observation_datetime'] = f"{date_str}T{time_str}"
            except:
                pass
        elif 'observation_date' in metadata:
            # DATE-OBS might contain full datetime
            metadata['observation_datetime'] = str(metadata['observation_date'])
        
        # Calculate plate scale if pixel scale available
        if 'pixel_scale_x' in metadata:
            # Convert to arcsec/pixel if in degrees
            scale = metadata['pixel_scale_x']
            if abs(scale) < 0.1:  # Likely in degrees
                metadata['plate_scale_arcsec_pixel'] = abs(scale) * 3600
            else:  # Already in arcsec
                metadata['plate_scale_arcsec_pixel'] = abs(scale)
        
        return metadata
    
    @staticmethod
    def load_sunpy_map(filepath):
        """
        Load image as SunPy Map for coordinate transformations
        
        Parameters:
        -----------
        filepath : str or Path
            Path to solar image
            
        Returns:
        --------
        solar_map : sunpy.map.Map or None
            SunPy Map object, None if SunPy not available or loading fails
        """
        
        if not SUNPY_AVAILABLE:
            print("Warning: SunPy not available, cannot create Map object")
            return None
        
        try:
            solar_map = sunpy.map.Map(str(filepath))
            return solar_map
        except Exception as e:
            print(f"Could not create SunPy Map: {e}")
            return None
    
    @staticmethod
    def standardize_image(image_data, target_size=None, normalize=True):
        """
        Standardize solar image format
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Input image data
        target_size : tuple, optional
            Target (height, width) for resizing
        normalize : bool
            Whether to normalize intensity values
            
        Returns:
        --------
        processed_image : numpy.ndarray
            Standardized image
        """
        
        if image_data is None:
            raise ValueError("Input image data is None")
        
        # Handle different data types
        if image_data.dtype == np.uint16:
            image_data = image_data.astype(np.float32)
        elif image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float32)
        
        # Handle NaN values
        if np.isnan(image_data).any():
            # Replace NaN with median of surrounding pixels
            mask = np.isnan(image_data)
            if mask.sum() > 0:
                image_data[mask] = ndimage.median_filter(image_data, size=3)[mask]
        
        # Normalize if requested
        if normalize:
            # Use robust normalization (remove outliers)
            finite_data = image_data[np.isfinite(image_data)]
            if len(finite_data) == 0:
                print("Warning: No finite values in image data")
                return image_data
            
            p1, p99 = np.percentile(finite_data, [1, 99])
            if p99 > p1:  # Avoid division by zero
                image_data = np.clip(image_data, p1, p99)
                image_data = (image_data - p1) / (p99 - p1)
        
        # Resize if requested
        if target_size is not None:
            if len(target_size) != 2:
                raise ValueError("target_size must be (height, width)")
            image_data = cv2.resize(image_data, target_size[::-1])  # cv2 uses (width, height)
        
        return image_data


class CoordinateTransformer:
    """
    Handle coordinate transformations for solar observations
    """
    
    def __init__(self, metadata=None, solar_map=None):
        """
        Initialize coordinate transformer
        
        Parameters:
        -----------
        metadata : dict, optional
            Image metadata from FITS header
        solar_map : sunpy.map.Map, optional
            SunPy Map object for transformations
        """
        
        self.metadata = metadata or {}
        self.solar_map = solar_map
        
        # Extract coordinate information
        self._setup_coordinate_system()
    
    def _setup_coordinate_system(self):
        """Setup coordinate system parameters"""
        
        # Default values - will be calculated dynamically when needed
        self.solar_radius_arcsec = None
        self.plate_scale = None
        self.solar_center_pixels = None
        
        # Try to get from metadata
        if self.metadata:
            if 'plate_scale_arcsec_pixel' in self.metadata:
                self.plate_scale = self.metadata['plate_scale_arcsec_pixel']
            
            if 'solar_radius_pixels' in self.metadata:
                solar_r_pix = self.metadata['solar_radius_pixels']
                # Calculate solar radius for observation time if available
                if 'observation_datetime' in self.metadata:
                    try:
                        obs_time = Time(self.metadata['observation_datetime'])
                        sun_coords = get_sun(obs_time)
                        self.solar_radius_arcsec = sun_coords.radius.to(u.arcsec).value
                    except:
                        self.solar_radius_arcsec = 959.63  # Fallback to average
                else:
                    self.solar_radius_arcsec = 959.63  # Fallback to average
                
                if self.plate_scale is None and self.solar_radius_arcsec is not None:
                    self.plate_scale = self.solar_radius_arcsec / solar_r_pix
            
            # Reference pixel often indicates solar center
            if 'reference_pixel_x' in self.metadata and 'reference_pixel_y' in self.metadata:
                self.solar_center_pixels = (
                    self.metadata['reference_pixel_x'],
                    self.metadata['reference_pixel_y']
                )
    
    def pixel_to_heliographic(self, x_pixels, y_pixels, observation_time):
        """
        Convert pixel coordinates to heliographic coordinates
        
        Parameters:
        -----------
        x_pixels, y_pixels : float or array
            Pixel coordinates
        observation_time : str or Time
            Observation time
            
        Returns:
        --------
        coordinates : SkyCoord
            Heliographic coordinates
        """
        
        if self.solar_map is not None and SUNPY_AVAILABLE:
            # Use SunPy Map for accurate transformation
            return self._pixel_to_heliographic_sunpy(x_pixels, y_pixels)
        else:
            # Use manual transformation
            return self._pixel_to_heliographic_manual(x_pixels, y_pixels, observation_time)
    
    def _pixel_to_heliographic_sunpy(self, x_pixels, y_pixels):
        """Use SunPy Map for coordinate transformation"""
        
        try:
            # Convert to world coordinates
            world_coords = self.solar_map.pixel_to_world(x_pixels * u.pixel, y_pixels * u.pixel)
            
            # Transform to heliographic
            heliographic_coords = world_coords.transform_to(frames.HeliographicStonyhurst)
            
            return heliographic_coords
            
        except Exception as e:
            print(f"Warning: SunPy coordinate transformation failed: {e}")
            # Fallback to manual transformation
            obs_time = self.solar_map.date if hasattr(self.solar_map, 'date') else Time.now()
            return self._pixel_to_heliographic_manual(x_pixels, y_pixels, obs_time)
    
    def _pixel_to_heliographic_manual(self, x_pixels, y_pixels, observation_time):
        """Manual coordinate transformation"""
        
        if self.plate_scale is None or self.solar_center_pixels is None:
            raise ValueError("Need plate scale and solar center for coordinate transformation")
        
        # Convert to Time object if needed
        if isinstance(observation_time, str):
            obs_time = Time(observation_time)
        else:
            obs_time = observation_time
        
        # Convert to angular coordinates
        center_x, center_y = self.solar_center_pixels
        dx_arcsec = (x_pixels - center_x) * self.plate_scale
        dy_arcsec = (y_pixels - center_y) * self.plate_scale
        
        try:
            if SUNPY_AVAILABLE:
                # Use proper Earth observer
                observer = get_earth(obs_time)
                
                # Create helioprojective coordinates
                helioprojective_coord = SkyCoord(
                    dx_arcsec * u.arcsec,
                    dy_arcsec * u.arcsec,
                    frame=frames.Helioprojective,
                    obstime=obs_time,
                    observer=observer
                )
                
                # Transform to heliographic
                heliographic_coord = helioprojective_coord.transform_to(frames.HeliographicStonyhurst)
                
            else:
                # Simplified transformation without SunPy
                # This is an approximation and less accurate
                r_arcsec = np.sqrt(dx_arcsec**2 + dy_arcsec**2)
                
                # Get solar radius for this time
                if self.solar_radius_arcsec is None:
                    try:
                        sun_coords = get_sun(obs_time)
                        solar_radius = sun_coords.radius.to(u.arcsec).value
                    except:
                        solar_radius = 959.63  # Fallback
                else:
                    solar_radius = self.solar_radius_arcsec
                
                # Simple projection (assumes small angles)
                if r_arcsec <= solar_radius:
                    # Convert to heliographic (simplified)
                    lon = np.degrees(np.arctan2(dx_arcsec, solar_radius)) * u.deg
                    lat = np.degrees(np.arcsin(dy_arcsec / solar_radius)) * u.deg
                    
                    heliographic_coord = SkyCoord(
                        lon, lat,
                        frame='heliographic_stonyhurst',
                        obstime=obs_time
                    )
                else:
                    # Point is beyond solar limb
                    heliographic_coord = SkyCoord(
                        np.nan * u.deg, np.nan * u.deg,
                        frame='heliographic_stonyhurst',
                        obstime=obs_time
                    )
            
            # Validate coordinates
            if hasattr(heliographic_coord, 'lon') and hasattr(heliographic_coord, 'lat'):
                lon_val = heliographic_coord.lon.to_value(u.deg)
                lat_val = heliographic_coord.lat.to_value(u.deg)
                
                # Check if coordinates are reasonable
                if not (-180 <= lon_val <= 180) or not (-90 <= lat_val <= 90):
                    print(f"Warning: Coordinates outside valid range: lon={lon_val:.2f}, lat={lat_val:.2f}")
            
            return heliographic_coord
            
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            # Return coordinates with NaN values
            return SkyCoord(
                np.nan * u.deg, np.nan * u.deg,
                frame='heliographic_stonyhurst',
                obstime=obs_time
            )
    
    def heliographic_to_pixel(self, longitude, latitude, observation_time):
        """
        Convert heliographic coordinates to pixel coordinates
        
        Parameters:
        -----------
        longitude, latitude : float or array
            Heliographic coordinates in degrees
        observation_time : str or Time
            Observation time
            
        Returns:
        --------
        x_pixels, y_pixels : float or array
            Pixel coordinates
        """
        
        if self.solar_map is not None and SUNPY_AVAILABLE:
            # Use SunPy Map
            try:
                heliographic_coord = SkyCoord(
                    longitude * u.deg,
                    latitude * u.deg,
                    frame=frames.HeliographicStonyhurst,
                    obstime=Time(observation_time)
                )
                
                pixel_coords = self.solar_map.world_to_pixel(heliographic_coord)
                return pixel_coords.x.value, pixel_coords.y.value
                
            except Exception as e:
                print(f"Warning: SunPy pixel conversion failed: {e}")
                # Fall through to manual method
        
        # Manual transformation
        if self.plate_scale is None or self.solar_center_pixels is None:
            raise ValueError("Need plate scale and solar center for coordinate transformation")
        
        # Convert to Time object if needed
        if isinstance(observation_time, str):
            obs_time = Time(observation_time)
        else:
            obs_time = observation_time
        
        try:
            if SUNPY_AVAILABLE:
                # Create heliographic coordinates
                heliographic_coord = SkyCoord(
                    longitude * u.deg,
                    latitude * u.deg,
                    frame=frames.HeliographicStonyhurst,
                    obstime=obs_time
                )
                
                # Transform to helioprojective
                observer = get_earth(obs_time)
                helioprojective_coord = heliographic_coord.transform_to(frames.Helioprojective(
                    observer=observer, obstime=obs_time
                ))
                
                # Convert to pixels
                center_x, center_y = self.solar_center_pixels
                x_pixels = center_x + helioprojective_coord.Tx.to(u.arcsec).value / self.plate_scale
                y_pixels = center_y + helioprojective_coord.Ty.to(u.arcsec).value / self.plate_scale
                
            else:
                # Simplified conversion without SunPy
                center_x, center_y = self.solar_center_pixels
                
                # Get solar radius
                if self.solar_radius_arcsec is None:
                    try:
                        sun_coords = get_sun(obs_time)
                        solar_radius = sun_coords.radius.to(u.arcsec).value
                    except:
                        solar_radius = 959.63
                else:
                    solar_radius = self.solar_radius_arcsec
                
                # Simple back-projection
                lon_rad = np.radians(longitude)
                lat_rad = np.radians(latitude)
                
                dx_arcsec = solar_radius * np.sin(lon_rad) * np.cos(lat_rad)
                dy_arcsec = solar_radius * np.sin(lat_rad)
                
                x_pixels = center_x + dx_arcsec / self.plate_scale
                y_pixels = center_y + dy_arcsec / self.plate_scale
            
            return x_pixels, y_pixels
            
        except Exception as e:
            print(f"Error in heliographic to pixel conversion: {e}")
            return np.nan, np.nan


class SolarDataDownloader:
    """
    Utility to download solar observation data
    """
    
    @staticmethod
    def download_hmi_images(start_time, end_time, cadence='12h', download_dir='solar_data'):
        """
        Download HMI continuum images from VSO
        
        Parameters:
        -----------
        start_time : str
            Start time (ISO format)
        end_time : str  
            End time (ISO format)
        cadence : str
            Time cadence (e.g., '12h', '1d')
        download_dir : str
            Download directory
            
        Returns:
        --------
        file_list : list
            List of downloaded files
        """
        
        if not SUNPY_NET_AVAILABLE:
            print("Error: SunPy.net not available. Cannot download data.")
            print("Install with: pip install sunpy[net]")
            return []
        
        download_dir = Path(download_dir)
        download_dir.mkdir(exist_ok=True)
        
        try:
            # Query for HMI continuum data
            result = Fido.search(
                a.Time(start_time, end_time),
                a.Instrument('HMI'),
                a.Physobs('intensity'),
                a.Sample(cadence)
            )
            
            print(f"Found {len(result)} files to download")
            
            if len(result) == 0:
                print("No files found for the specified criteria")
                return []
            
            # Download files
            downloaded_files = Fido.fetch(result, path=download_dir)
            
            print(f"Downloaded {len(downloaded_files)} files to {download_dir}")
            return downloaded_files
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return []
    
    @staticmethod
    def create_time_series(start_time, end_time, cadence_hours=12):
        """
        Create a time series for observations
        
        Parameters:
        -----------
        start_time : str
            Start time (ISO format)
        end_time : str
            End time (ISO format)
        cadence_hours : float
            Time between observations in hours
            
        Returns:
        --------
        time_series : list
            List of datetime strings
        """
        
        if cadence_hours <= 0:
            raise ValueError("cadence_hours must be positive")
        
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            return []
        
        if start_dt >= end_dt:
            raise ValueError("start_time must be before end_time")
        
        time_series = []
        current_time = start_dt
        
        while current_time <= end_dt:
            time_series.append(current_time.isoformat())
            current_time += timedelta(hours=cadence_hours)
        
        return time_series


class QualityAssessment:
    """
    Assess quality of solar images and sunspot detections
    """
    
    @staticmethod
    def assess_image_quality(image_data, solar_mask=None):
        """
        Assess overall image quality
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Solar image data
        solar_mask : numpy.ndarray, optional
            Solar disk mask
            
        Returns:
        --------
        quality_metrics : dict
            Quality assessment metrics
        """
        
        if image_data is None:
            raise ValueError("image_data cannot be None")
        
        metrics = {}
        
        if solar_mask is not None:
            if solar_mask.shape != image_data.shape:
                raise ValueError("solar_mask shape must match image_data shape")
            data_region = image_data[solar_mask]
        else:
            data_region = image_data.flatten()
        
        if len(data_region) == 0:
            print("Warning: No data in specified region")
            return {'quality_score': 0.0, 'error': 'No data in region'}
        
        # Basic statistics
        metrics['mean_intensity'] = np.mean(data_region)
        metrics['std_intensity'] = np.std(data_region)
        metrics['dynamic_range'] = np.ptp(data_region)  # Peak-to-peak
        
        # Signal-to-noise ratio estimation
        # Use Laplacian variance as noise estimate
        try:
            laplacian = cv2.Laplacian(image_data.astype(np.float32), cv2.CV_32F)
            if solar_mask is not None:
                noise_estimate = np.var(laplacian[solar_mask])
            else:
                noise_estimate = np.var(laplacian)
            
            metrics['noise_estimate'] = noise_estimate
            metrics['snr_estimate'] = metrics['mean_intensity']**2 / noise_estimate if noise_estimate > 0 else np.inf
            
        except Exception as e:
            print(f"Warning: Could not calculate noise estimate: {e}")
            metrics['noise_estimate'] = 0.0
            metrics['snr_estimate'] = np.inf
        
        # Contrast metrics
        if metrics['std_intensity'] > 0:
            metrics['rms_contrast'] = np.sqrt(np.mean((data_region - metrics['mean_intensity'])**2))
            range_sum = np.max(data_region) + np.min(data_region)
            if range_sum > 0:
                metrics['michelson_contrast'] = (np.max(data_region) - np.min(data_region)) / range_sum
            else:
                metrics['michelson_contrast'] = 0.0
        else:
            metrics['rms_contrast'] = 0.0
            metrics['michelson_contrast'] = 0.0
        
        # Overall quality score (0-1)
        snr_score = min(1.0, metrics['snr_estimate'] / 100) if np.isfinite(metrics['snr_estimate']) else 0.0
        contrast_score = min(1.0, metrics['michelson_contrast'] * 10)
        quality_score = snr_score * contrast_score
        metrics['quality_score'] = quality_score
        
        return metrics
    
    @staticmethod
    def assess_sunspot_detection(sunspots, image_data, solar_mask):
        """
        Assess quality of sunspot detection
        
        Parameters:
        -----------
        sunspots : list
            Detected sunspots
        image_data : numpy.ndarray
            Solar image data
        solar_mask : numpy.ndarray
            Solar disk mask
            
        Returns:
        --------
        detection_metrics : dict
            Detection quality metrics
        """
        
        if image_data is None or solar_mask is None:
            raise ValueError("image_data and solar_mask cannot be None")
        
        if sunspots is None:
            sunspots = []
        
        metrics = {
            'num_sunspots': len(sunspots),
            'total_sunspot_area': sum(spot['area'] for spot in sunspots if 'area' in spot),
            'mean_sunspot_size': np.mean([spot['area'] for spot in sunspots if 'area' in spot]) if sunspots else 0,
            'size_distribution': [spot['area'] for spot in sunspots if 'area' in spot],
            'intensity_distribution': [spot['mean_intensity'] for spot in sunspots if 'mean_intensity' in spot]
        }
        
        # Calculate sunspot filling factor
        solar_disk_area = np.sum(solar_mask)
        if solar_disk_area > 0:
            metrics['filling_factor'] = metrics['total_sunspot_area'] / solar_disk_area
        else:
            metrics['filling_factor'] = 0.0
        
        # Assess detection completeness (heuristic)
        try:
            if len(sunspots) > 0:
                mean_background = np.mean(image_data[solar_mask])
                std_background = np.std(image_data[solar_mask])
                dark_threshold = mean_background - 2 * std_background
                dark_pixel_fraction = np.sum((image_data < dark_threshold) & solar_mask) / solar_disk_area
                
                if dark_pixel_fraction > 0:
                    detected_fraction = metrics['total_sunspot_area'] / (dark_pixel_fraction * solar_disk_area)
                    metrics['detection_completeness_estimate'] = min(1.0, detected_fraction)
                else:
                    metrics['detection_completeness_estimate'] = 1.0
            else:
                # No sunspots detected - check if there should be any
                mean_background = np.mean(image_data[solar_mask])
                std_background = np.std(image_data[solar_mask])
                dark_threshold = mean_background - 2 * std_background
                dark_pixels = np.sum((image_data < dark_threshold) & solar_mask)
                
                metrics['detection_completeness_estimate'] = 1.0 if dark_pixels == 0 else 0.0
                
        except Exception as e:
            print(f"Warning: Could not calculate detection completeness: {e}")
            metrics['detection_completeness_estimate'] = 0.5  # Unknown
        
        return metrics


def create_analysis_config_template():
    """
    Create a template configuration file for solar analysis
    Updated to match the pipeline expectations with flat structure
    
    Returns:
    --------
    config : dict
        Configuration template
    """
    
    config = {
        "data_sources": {
            "local_files": {
                "image_directory": "/path/to/solar/images",
                "file_pattern": "*.fits",
                "time_from_filename": True,
                "filename_time_format": "%Y%m%d_%H%M%S"
            },
            "download": {
                "enabled": False,
                "instrument": "HMI",
                "data_type": "continuum",
                "start_time": "2023-10-15T00:00:00",
                "end_time": "2023-10-22T00:00:00",
                "cadence": "12h"
            }
        },
        
        "preprocessing": {
            "image_standardization": {
                "normalize": True,
                "target_size": None,
                "remove_outliers": True
            },
            "coordinate_system": {
                "use_sunpy_maps": True,
                "manual_plate_scale": None,
                "manual_solar_center": None
            }
        },
        
        # Flattened detection config to match pipeline expectations
        "detection": {
            "disk_method": "hough",  # Pipeline expects this key directly
            "min_sunspot_area": 10,  # Pipeline expects this key directly
            "intensity_threshold": None,  # Pipeline expects this key directly
            "limb_darkening_correction": True,  # Pipeline expects this key directly
            
            # Additional parameters for disk detection
            "edge_detection_sigma": 1.0,
            "hough_params": {
                "param1": 50,
                "param2": 30,
                "min_radius_fraction": 0.3,
                "max_radius_fraction": 0.6
            },
            
            # Additional parameters for sunspot detection
            "intensity_threshold_sigma": 2.0,
            "morphology_operations": {
                "remove_small_objects": True,
                "fill_holes": True
            }
        },
        
        # Flattened tracking config to match pipeline expectations
        "tracking": {
            "max_distance_deg": 15.0,  # Pipeline expects this key directly
            "max_time_days": 7.0,      # Pipeline expects this key directly  
            "min_track_length": 2      # Pipeline expects this key directly
        },
        
        # Flattened analysis config to match pipeline expectations
        "analysis": {
            "fit_differential_rotation": True,  # Pipeline expects this key directly
            "export_results": True,             # Pipeline expects this key directly
            "create_plots": True,               # Pipeline expects this key directly
            "exclude_high_latitude": 60.0,
            "minimum_tracks_for_fit": 3
        },
        
        # Flattened output config to match pipeline expectations  
        "output": {
            "results_dir": "solar_analysis_results",  # Pipeline expects this key directly
            "save_intermediate": True,                # Pipeline expects this key directly
            "plot_format": "png",                     # Pipeline expects this key directly
            
            # Additional output settings
            "create_summary_plots": True,
            "create_animation": False,
            "plot_individual_detections": True,
            "plot_tracking_results": True,
            "data_format": "csv",
            "export_json": True
        },
        
        "quality_control": {
            "image_quality": {
                "min_snr": 10.0,
                "min_contrast": 0.1,
                "min_quality_score": 0.3
            },
            "detection_quality": {
                "max_sunspot_size": 1000,
                "min_detection_completeness": 0.5
            }
        }
    }
    
    return config


def save_config_template(filename='solar_analysis_config.json'):
    """Save configuration template to file"""
    
    config = create_analysis_config_template()
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration template saved to {filename}")
        print("Edit this file to customize your analysis parameters")
        
    except Exception as e:
        print(f"Error saving configuration template: {e}")


def validate_dependencies():
    """
    Validate that required dependencies are available
    
    Returns:
    --------
    status : dict
        Dictionary with dependency availability status
    """
    
    status = {
        'numpy': False,
        'matplotlib': False,
        'astropy': False,
        'opencv': False,
        'scipy': False,
        'sunpy': SUNPY_AVAILABLE,
        'sunpy_net': SUNPY_NET_AVAILABLE
    }
    
    try:
        import numpy
        status['numpy'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        status['matplotlib'] = True
    except ImportError:
        pass
    
    try:
        import astropy
        status['astropy'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        status['opencv'] = True
    except ImportError:
        pass
    
    try:
        import scipy
        status['scipy'] = True
    except ImportError:
        pass
    
    return status


def print_dependency_status():
    """Print the status of all dependencies"""
    
    status = validate_dependencies()
    
    print("Solar Analysis Utilities - Dependency Status")
    print("=" * 45)
    
    for package, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"{symbol} {package:<12} {'Available' if available else 'Missing'}")
    
    missing = [pkg for pkg, avail in status.items() if not avail]
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("\nAll dependencies are available!")


if __name__ == "__main__":
    # Check dependencies first
    print_dependency_status()
    print()
    
    # Create example configuration file
    save_config_template()
    
    # Example usage of utilities
    print("\nExample utility usage:")
    print("1. Load FITS image:")
    print("   loader = SolarImageLoader()")
    print("   data, header, metadata = loader.load_fits_with_header('image.fits')")
    
    print("\n2. Setup coordinate transformer:")
    print("   transformer = CoordinateTransformer(metadata)")
    print("   coords = transformer.pixel_to_heliographic(x, y, obs_time)")
    
    if SUNPY_NET_AVAILABLE:
        print("\n3. Download solar data:")
        print("   downloader = SolarDataDownloader()")
        print("   files = downloader.download_hmi_images('2023-10-15', '2023-10-22')")
    else:
        print("\n3. Download solar data: (Requires sunpy[net])")
        print("   pip install sunpy[net]")
    
    print("\n4. Assess image quality:")
    print("   qa = QualityAssessment()")
    print("   quality = qa.assess_image_quality(image_data, solar_mask)")
    
    print("\n5. Test coordinate transformation:")
    print("   # Test with synthetic data")
    print("   test_image = np.random.rand(512, 512)")
    print("   test_metadata = {'plate_scale_arcsec_pixel': 0.6, 'reference_pixel_x': 256, 'reference_pixel_y': 256}")
    print("   transformer = CoordinateTransformer(test_metadata)")
    print("   coords = transformer.pixel_to_heliographic(300, 200, '2023-10-15T12:00:00')")
    print("   print(f'Coordinates: {coords.lon:.2f}, {coords.lat:.2f}')")