# Solar Differential Rotation Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SunPy](https://img.shields.io/badge/SunPy-Compatible-orange.svg)](https://sunpy.org)

A comprehensive Python pipeline for detecting, tracking, and analyzing sunspots to measure solar differential rotation from time-series solar images.

## 🌟 Overview

This pipeline automatically:
- Detects solar disk and sunspots in solar images
- Tracks sunspots across multiple observations
- Measures rotation rates at different solar latitudes
- Fits differential rotation parameters
- Compares results with literature values

## 📁 Project Structure

```
solar-rotation-pipeline/
├── solar_pipeline.py          # Main pipeline orchestrator
├── sunspot_detection.py       # Solar disk and sunspot detection
├── sunspot_tracking.py        # Sunspot tracking and rotation analysis
├── solar_utilities.py         # Utility functions (mostly unused)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── examples/
    ├── sample_config.json      # Example configuration
    └── sample_images/          # Example solar images
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Required packages:

```bash
pip install numpy matplotlib astropy sunpy opencv-python scikit-image scipy pandas
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For enhanced functionality:
```bash
# For downloading solar data
pip install sunpy[net]

# For additional image processing
pip install pillow
```

## 🚀 Quick Start

### 1. Create Configuration Template
```bash
python solar_pipeline.py --create-config
```
This creates `solar_analysis_config.json` with default parameters.

### 2. Basic Usage with Local Images
```bash
python solar_pipeline.py \
  --images image1.fits image2.fits image3.fits \
  --times "2023-10-15T12:00" "2023-10-17T12:00" "2023-10-19T12:00"
```

### 3. Advanced Usage with Configuration File
```bash
python solar_pipeline.py --config solar_analysis_config.json
```

## 📊 Expected Results

### Console Output
```
Processing 3 images...
Processing image 1/3: image1.fits
Solar disk detected: center=(512.1, 512.3), radius=450.2 pixels
  Found 5 sunspots
Processing image 2/3: image2.fits
  Found 4 sunspots
Processing image 3/3: image3.fits
  Found 6 sunspots
Successfully processed 3/3 images

Starting tracking analysis...
Matching sunspots across observations...
Created 3 sunspot tracks
Calculating rotation rates...
Fitting differential rotation law...

Differential Rotation Fit Results:
ω(θ) = 14.234 + -2.156*sin²(θ) + -1.345*sin⁴(θ) deg/day
Equatorial rotation rate: 14.234 ± 0.123 deg/day
Equatorial period: 25.31 days
R-squared: 0.8456

Analysis completed successfully!
Found 3 sunspot tracks
```

### Generated Files

The pipeline creates a `solar_analysis_results/` directory with:

#### 📄 **Core Results**
- `rotation_analysis.csv` - Detailed rotation measurements
- `differential_rotation_fit.json` - Mathematical fit parameters
- `sunspot_tracks.json` - Track summary information
- `analysis_report.txt` - Human-readable summary report

#### 📈 **Data Files**
- `image_000_results.json` - Individual image processing results
- `image_001_results.json` - (one per input image)
- `image_002_results.json`

#### 🖼️ **Visualizations**
The pipeline automatically displays plots showing:
- Original images with detected solar disks
- Sunspot detection masks
- Sunspot longitude evolution over time
- Rotation rate vs. latitude scatter plot
- Fitted differential rotation curve
- Track quality assessment
- Fit residuals analysis

### Sample Data Structure

#### Rotation Analysis CSV
```csv
track_id,mean_latitude,rotation_rate_deg_day,observations_count,time_span_days
0,15.2,13.8,3,2.0
1,-8.7,14.1,3,2.0
2,22.4,13.2,2,1.0
```

#### Differential Rotation Fit JSON
```json
{
  "A": 14.234,
  "B": -2.156,
  "C": -1.345,
  "A_error": 0.123,
  "B_error": 0.445,
  "C_error": 0.234,
  "r_squared": 0.8456,
  "equatorial_period": 25.31
}
```

## ⚙️ Configuration Options

### Detection Parameters
```json
{
  "detection": {
    "disk_method": "hough",           // or "edge"
    "min_sunspot_area": 10,           // minimum pixels
    "intensity_threshold": null,      // auto-calculated if null
    "limb_darkening_correction": true
  }
}
```

### Tracking Parameters
```json
{
  "tracking": {
    "max_distance_deg": 15.0,    // max distance for matching
    "max_time_days": 7.0,        // max time gap
    "min_track_length": 2        // minimum observations per track
  }
}
```

### Output Settings
```json
{
  "output": {
    "results_dir": "solar_analysis_results",
    "save_intermediate": true,
    "plot_format": "png"
  }
}
```

## 📸 Input Image Requirements

### Recommended Format
- **FITS files** from solar observatories (SDO/HMI, SOHO, etc.)
- **Time series**: 2-10 images spanning 1-7 days
- **Cadence**: 12-24 hours between observations
- **Quality**: Clear sunspot features visible

### Supported Formats
- `.fits` - Preferred (includes astronomical metadata)
- `.jpg/.png` - Supported but limited coordinate accuracy

### Image Specifications
- **Size**: 512x512 to 4096x4096 pixels
- **Content**: Full solar disk visible
- **Features**: At least 1-2 sunspots for tracking

## 🎯 Success Scenarios

### Excellent Results (R² > 0.9)
- 5+ images over 3-5 days
- Multiple large sunspot groups
- High-quality FITS data
- 5+ successful tracks

### Good Results (R² > 0.7)
- 3-4 images over 2-3 days  
- 2-3 visible sunspots
- 3+ successful tracks

### Limited Results
- 2-3 images
- 1-2 sunspots
- Basic rotation rate measurements

## ⚠️ Common Issues & Solutions

### No Sunspots Detected
**Symptoms**: "Found 0 sunspots" for all images
**Solutions**:
- Lower `min_sunspot_area` parameter
- Adjust `intensity_threshold` 
- Check if images actually contain sunspots
- Try different `disk_method`

### Tracking Failed
**Symptoms**: "Created 0 sunspot tracks"
**Solutions**:
- Increase `max_distance_deg` parameter
- Reduce time gaps between images
- Ensure sunspots persist across observations

### Coordinate Conversion Errors
**Symptoms**: Warnings about coordinate transformation
**Solutions**:
- Use FITS files with proper WCS headers
- Check observation times are correct
- Verify solar disk detection accuracy

### Poor Fit Quality (Low R²)
**Solutions**:
- Increase number of tracked sunspots
- Extend observation time span
- Use higher quality images
- Filter tracks at extreme latitudes

## 📚 Scientific Background

### Differential Rotation Law
The solar rotation rate varies with latitude according to:

```
ω(θ) = A + B·sin²(θ) + C·sin⁴(θ)
```

Where:
- **A**: Equatorial rotation rate (~14.7 deg/day)
- **B**: First-order latitude term (~-2.4 deg/day)  
- **C**: Second-order latitude term (~-1.8 deg/day)
- **θ**: Heliographic latitude

### Literature Values (Snodgrass & Ulrich 1990)
- A = 14.713 ± 0.0491 deg/day
- B = -2.396 ± 0.188 deg/day
- C = -1.787 ± 0.253 deg/day

## 🔬 Advanced Usage

### Programmatic Interface
```python
from solar_pipeline import SolarAnalysisPipeline

# Custom configuration
config = {
    'detection': {
        'min_sunspot_area': 5,
        'disk_method': 'hough'
    },
    'tracking': {
        'max_distance_deg': 20.0
    }
}

# Initialize pipeline
pipeline = SolarAnalysisPipeline(config)

# Process images
image_files = ["img1.fits", "img2.fits", "img3.fits"]
times = ["2023-10-15T12:00", "2023-10-17T12:00", "2023-10-19T12:00"]

success = pipeline.process_image_sequence(image_files, times)

if success:
    results = pipeline.run_tracking_analysis()
    pipeline.create_summary_report()
    
    # Access results
    if results:
        fit_params = results['fit_parameters']
        rotation_data = results['rotation_data']
        tracks = results['tracks']
```

### Batch Processing
```python
# Process multiple datasets
datasets = [
    {
        'images': ['day1_img1.fits', 'day1_img2.fits'],
        'times': ['2023-10-15T12:00', '2023-10-17T12:00'],
        'output_dir': 'results_dataset1'
    },
    # ... more datasets
]

for dataset in datasets:
    config['output']['results_dir'] = dataset['output_dir']
    pipeline = SolarAnalysisPipeline(config)
    # ... process dataset
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check this README for common solutions
2. Review the generated `analysis_report.txt` for diagnostic information
3. Enable debug logging by modifying the logging level in `solar_pipeline.py`

## 🙏 Acknowledgments

- Solar Dynamics Observatory (SDO) for high-quality solar data
- SunPy community for solar physics tools
- Scientific references: Snodgrass & Ulrich (1990), Beck (2000)

## 📖 Example Command Reference

```bash
# Create config template
python solar_pipeline.py --create-config

# Basic analysis
python solar_pipeline.py --images *.fits --times "2023-10-15T12:00" "2023-10-17T12:00"

# With custom output directory
python solar_pipeline.py --config config.json --output-dir my_analysis

# Help
python solar_pipeline.py --help
```

---

**Happy solar rotation analysis!** 🌞
