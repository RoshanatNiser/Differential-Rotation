# API Reference

Complete API documentation for the Solar Differential Rotation Analysis Pipeline.

## 📚 Module Overview

```python
solar-rotation-pipeline/
├── solar_pipeline.py          # Main pipeline class
├── sunspot_detection.py       # Detection classes  
├── sunspot_tracking.py        # Tracking and analysis classes
└── solar_utilities.py         # Utility functions
```

## 🎯 Main Pipeline Class

### `SolarAnalysisPipeline`

The main orchestrator class that coordinates the entire analysis workflow.

```python
from solar_pipeline import SolarAnalysisPipeline
```

#### Constructor

```python
SolarAnalysisPipeline(config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary. If None, uses default settings.

**Example:**
```python
config = {
    'detection': {'min_sunspot_area': 10},
    'tracking': {'max_distance_deg': 15.0}
}
pipeline = SolarAnalysisPipeline(config)
```

#### Methods

##### `process_image_sequence(image_paths, observation_times)`

Process a sequence of solar images for sunspot detection.

**Parameters:**
- `image_paths` (list): List of paths to solar images
- `observation_times` (list): List of observation times (ISO format strings)

**Returns:**
- `bool`: True if processing successful, False otherwise

**Example:**
```python
success = pipeline.process_image_sequence(
    ["img1.fits", "img2.fits"],
    ["2023-10-15T12:00:00", "2023-10-17T12:00:00"]
)
```

##### `run_tracking_analysis()`

Run sunspot tracking and differential rotation analysis.

**Returns:**
- `dict`: Analysis results with keys:
  - `rotation_data`: List of rotation measurements
  - `fit_parameters`: Differential rotation fit parameters
  - `tracks`: List of sunspot tracks
  - `results_dataframe`: Pandas DataFrame with results

**Example:**
```python
results = pipeline.run_tracking_analysis()
if results:
    fit_params = results['fit_parameters']
    print(f"Equatorial rate: {fit_params['A']:.3f} deg/day")
```

##### `create_summary_report()`

Generate a human-readable summary report.

**Returns:**
- None (writes report to file)

## 🔍 Detection Module

### `SolarDiskDetector`

Detects solar disk in images using various methods.

```python
from sunspot_detection import SolarDiskDetector
```

#### Methods

##### `detect_solar_disk(image_data, method='hough')`

Detect solar disk boundary.

**Parameters:**
- `image_data` (np.ndarray): Input solar image
- `method` (str): Detection method ('hough' or 'edge')

**Returns:**
- `tuple`: (center, radius, mask)
  - `center` (tuple): (x, y) center coordinates
  - `radius` (float): Solar disk radius in pixels
  - `mask` (np.ndarray): Binary mask of solar disk

**Example:**
```python
detector = SolarDiskDetector()
center, radius, mask = detector.detect_solar_disk(image_data, method='hough')
```

##### `load_image(filepath)`

Load image from file (FITS or standard formats).

**Parameters:**
- `filepath` (str): Path to image file

**Returns:**
- `tuple`: (image_data, header)

### `SunspotDetector`

Detects and analyzes sunspots within solar disk.

```python
from sunspot_detection import SunspotDetector
```

#### Constructor

```python
SunspotDetector(solar_center, solar_radius)
```

**Parameters:**
- `solar_center` (tuple): (x, y) solar disk center
- `solar_radius` (float): Solar disk radius in pixels

#### Methods

##### `detect_sunspots(image_data, solar_mask, min_area=10, intensity_threshold=None)`

Detect sunspots in solar image.

**Parameters:**
- `image_data` (np.ndarray): Input solar image
- `solar_mask` (np.ndarray): Binary mask of solar disk
- `min_area` (int): Minimum sunspot area in pixels
- `intensity_threshold` (float, optional): Custom detection threshold

**Returns:**
- `tuple`: (sunspots, sunspot_mask)
  - `sunspots` (list): List of sunspot dictionaries
  - `sunspot_mask` (np.ndarray): Binary mask of detected sunspots

**Sunspot Dictionary Structure:**
```python
{
    'centroid': (x, y),              # Centroid position
    'area': float,                   # Area in pixels
    'mean_intensity': float,         # Average intensity
    'heliographic_lon': Quantity,    # Longitude (if converted)
    'heliographic_lat': Quantity,    # Latitude (if converted)
    'bbox': tuple,                   # Bounding box
    'equivalent_diameter': float,    # Equivalent circular diameter
    'eccentricity': float           # Shape eccentricity
}
```

##### `pixel_to_heliographic(pixel_coords, observation_time, plate_scale=None)`

Convert pixel coordinates to heliographic coordinates.

**Parameters:**
- `pixel_coords` (tuple or list): (x, y) coordinates or list of coordinates
- `observation_time` (str or Time): Observation time
- `plate_scale` (float, optional): Arcseconds per pixel

**Returns:**
- `list`: List of SkyCoord objects in heliographic coordinates

## 📊 Tracking Module

### `SunspotTracker`

Tracks sunspots across multiple observations.

```python
from sunspot_tracking import SunspotTracker
```

#### Methods

##### `add_observation(sunspots, observation_time, image_info)`

Add observation to tracking system.

**Parameters:**
- `sunspots` (list): List of detected sunspots
- `observation_time` (str or Time): Observation time
- `image_info` (dict): Image metadata

##### `match_sunspots(max_distance_deg=15.0, max_time_days=7.0)`

Match sunspots across observations to create tracks.

**Parameters:**
- `max_distance_deg` (float): Maximum matching distance in degrees
- `max_time_days` (float): Maximum time gap for matching

##### `calculate_rotation_rates()`

Calculate rotation rates for tracked sunspots.

**Returns:**
- `list`: List of rotation data dictionaries

**Rotation Data Structure:**
```python
{
    'track_id': int,                    # Unique track identifier
    'mean_latitude': float,             # Average latitude (degrees)
    'rotation_rate_deg_day': float,     # Rotation rate (deg/day)
    'observations_count': int,          # Number of observations
    'time_span_days': float,           # Time span of track
    'longitude_change': float          # Total longitude change
}
```

### `DifferentialRotationAnalyzer`

Complete analysis pipeline for differential rotation.

```python
from sunspot_tracking import DifferentialRotationAnalyzer
```

#### Methods

##### `analyze_image_sequence(image_data_list, observation_times)`

Analyze complete image sequence.

**Parameters:**
- `image_data_list` (list): List of (image_path, sunspots, disk_info) tuples
- `observation_times` (list): List of observation times

**Returns:**
- `dict`: Complete analysis results

##### `compare_with_literature(fit_params)`

Compare fitted parameters with literature values.

**Parameters:**
- `fit_params` (dict): Fitted differential rotation parameters

## 🛠️ Utility Functions

### Configuration

```python
from solar_utilities import create_analysis_config_template
```

##### `create_analysis_config_template()`

Create default configuration template.

**Returns:**
- `dict`: Configuration template with all parameters

### Image Loading

```python
from solar_utilities import SolarImageLoader
```

##### `SolarImageLoader.load_fits_with_header(filepath)`

Load FITS file with comprehensive metadata extraction.

**Parameters:**
- `filepath` (str or Path): Path to FITS file

**Returns:**
- `tuple`: (image_data, header, metadata)

### Coordinate Transformation

```python
from solar_utilities import CoordinateTransformer
```

##### `CoordinateTransformer(metadata=None, solar_map=None)`

Initialize coordinate transformer.

**Parameters:**
- `metadata` (dict, optional): Image metadata
- `solar_map` (sunpy.map.Map, optional): SunPy Map object

### Quality Assessment

```python
from solar_utilities import QualityAssessment
```

##### `QualityAssessment.assess_image_quality(image_data, solar_mask=None)`

Assess overall image quality.

**Parameters:**
- `image_data` (np.ndarray): Solar image data
- `solar_mask` (np.ndarray, optional): Solar disk mask

**Returns:**
- `dict`: Quality metrics including SNR, contrast, quality score

## 📈 Data Structures

### Configuration Dictionary

Complete configuration structure:

```python
config = {
    "detection": {
        "disk_method": "hough",              # "hough" or "edge"
        "min_sunspot_area": 10,              # Minimum area in pixels
        "intensity_threshold": None,          # Auto-calculated if None
        "limb_darkening_correction": True     # Apply correction
    },
    "tracking": {
        "max_distance_deg": 15.0,            # Max matching distance
        "max_time_days": 7.0,                # Max time gap
        "min_track_length": 2                # Min observations per track
    },
    "analysis": {
        "fit_differential_rotation": True,    # Enable rotation fitting
        "export_results": True,               # Save results to files
        "create_plots": True                  # Generate visualizations
    },
    "output": {
        "results_dir": "solar_analysis_results",  # Output directory
        "save_intermediate": True,                # Save per-image results
        "plot_format": "png"                     # Plot file format
    }
}
```

### Analysis Results Dictionary

Structure of complete analysis results:

```python
results = {
    "rotation_data": [                    # List of rotation measurements
        {
            "track_id": 0,
            "mean_latitude": 15.2,
            "rotation_rate_deg_day": 13.8,
            "observations_count": 3,
            "time_span_days": 2.0
        }
    ],
    "fit_parameters": {                   # Differential rotation fit
        "A": 14.234,                      # Equatorial rate (deg/day)
        "B": -2.156,                      # First-order term
        "C": -1.345,                      # Second-order term
        "A_error": 0.123,                 # Parameter uncertainties
        "B_error": 0.445,
        "C_error": 0.234,
        "r_squared": 0.8456,              # Fit quality
        "equatorial_period": 25.31        # Equatorial period (days)
    },
    "tracks": [                           # Complete track information
        {
            "track_id": 0,
            "observations": [              # List of (time, sunspot) pairs
                (Time('2023-10-15T12:00:00'), sunspot_dict)
            ],
            "active": False               # Whether track is still active
        }
    ],
    "results_dataframe": pandas.DataFrame  # Tabular results
}
```

## ⚠️ Error Handling

### Common Exceptions

The pipeline raises specific exceptions for different error conditions:

```python
# Image loading errors
FileNotFoundError: Image file not found
ValueError: Invalid image format or corrupted data

# Detection errors  
ValueError: Could not detect solar disk
RuntimeError: Sunspot detection failed

# Tracking errors
ValueError: Insufficient observations for tracking
RuntimeError: Coordinate transformation failed

# Analysis errors
ValueError: Insufficient tracks for differential rotation fit
RuntimeError: Fit convergence failed
```

### Error Handling Example

```python
try:
    pipeline = SolarAnalysisPipeline(config)
    success = pipeline.process_image_sequence(images, times)
    
    if success:
        results = pipeline.run_tracking_analysis()
        if results is None:
            print("Analysis failed - check intermediate results")
    else:
        print("Image processing failed")
        
except FileNotFoundError as e:
    print(f"Image file not found: {e}")
except ValueError as e:
    print(f"Invalid input parameters: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔧 Advanced Usage

### Custom Detection Algorithm

```python
from sunspot_detection import SolarDiskDetector

class CustomSolarDetector(SolarDiskDetector):
    def detect_solar_disk(self, image_data, method='custom'):
        # Your custom detection algorithm
        center, radius, mask = self.custom_detection(image_data)
        return center, radius, mask
```

### Custom Tracking Metrics

```python
from sunspot_tracking import SunspotTracker

tracker = SunspotTracker()
# Add custom distance metric
def custom_distance(coord1, coord2):
    # Your custom distance calculation
    return distance

# Use in matching
tracker.match_sunspots(distance_func=custom_distance)
```

## 📊 Performance Considerations

### Memory Usage
- Large images (>2048x2048): ~100MB memory per image
- Typical analysis (5 images): ~500MB peak memory usage
- Enable `save_intermediate=False` for memory-constrained systems

### Processing Time
- Single image processing: 10-60 seconds depending on size
- Complete analysis (5 images): 2-10 minutes
- Tracking and fitting: Usually <1 minute

### Optimization Tips
- Use appropriate `target_size` for large images
- Disable unnecessary visualizations for batch processing
- Use `min_track_length >= 3` for
