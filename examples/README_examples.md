# Solar Pipeline Examples

This directory contains example usage patterns and sample configurations for the Solar Differential Rotation Analysis Pipeline.

## 📁 Files Overview

- **`sample_config.json`** - Complete configuration template with all options
- **`sample_analysis.py`** - Python examples showing different usage patterns
- **`README_examples.md`** - This documentation file

## 🚀 Quick Start Examples

### 1. Basic Command Line Usage

```bash
# Create configuration template
python ../solar_pipeline.py --create-config

# Basic analysis with 3 images
python ../solar_pipeline.py \
  --images solar1.fits solar2.fits solar3.fits \
  --times "2023-10-15T12:00" "2023-10-17T12:00" "2023-10-19T12:00"

# Using configuration file
python ../solar_pipeline.py --config sample_config.json
```

### 2. Python Script Usage

```python
from solar_pipeline import SolarAnalysisPipeline

# Simple usage
pipeline = SolarAnalysisPipeline()
images = ["img1.fits", "img2.fits", "img3.fits"]
times = ["2023-10-15T12:00", "2023-10-17T12:00", "2023-10-19T12:00"]

success = pipeline.process_image_sequence(images, times)
if success:
    results = pipeline.run_tracking_analysis()
```

## 📊 Example Scenarios

### Scenario 1: High Solar Activity Period
```json
{
  "detection": {
    "min_sunspot_area": 8,
    "intensity_threshold": null
  },
  "tracking": {
    "max_distance_deg": 12.0,
    "max_time_days": 3.0
  }
}
```
**Best for**: Periods with many large sunspots, short observation cadence

### Scenario 2: Low Solar Activity Period  
```json
{
  "detection": {
    "min_sunspot_area": 3,
    "intensity_threshold": -2.5
  },
  "tracking": {
    "max_distance_deg": 20.0,
    "max_time_days": 7.0
  }
}
```
**Best for**: Solar minimum periods, few small sunspots

### Scenario 3: High-Precision Analysis
```json
{
  "detection": {
    "min_sunspot_area": 10,
    "limb_darkening_correction": true
  },
  "tracking": {
    "min_track_length": 4
  },
  "analysis": {
    "minimum_tracks_for_fit": 8
  }
}
```
**Best for**: Research-quality measurements, longer time series

## 🎯 Expected Results by Scenario

### Excellent Results (R² > 0.9)
- **Dataset**: 7+ images over 5-7 days
- **Sunspots**: 3+ active regions with multiple spots
- **Cadence**: 12-24 hour intervals
- **Result**: 8+ tracks, precise differential rotation parameters

### Good Results (R² > 0.7)
- **Dataset**: 4-6 images over 3-5 days  
- **Sunspots**: 2-3 active regions
- **Cadence**: 24-48 hour intervals
- **Result**: 4-7 tracks, reliable rotation measurements

### Moderate Results (R² > 0.5)
- **Dataset**: 3-4 images over 2-3 days
- **Sunspots**: 1-2 persistent active regions
- **Cadence**: 24 hour intervals
- **Result**: 2-4 tracks, basic rotation trend

## 🔧 Configuration Guide

### Detection Parameters

| Parameter | Description | Typical Range | Notes |
|-----------|-------------|---------------|--------|
| `min_sunspot_area` | Minimum sunspot size (pixels) | 5-20 | Lower for small spots |
| `intensity_threshold` | Detection threshold | null (auto) | null = auto-calculate |
| `disk_method` | Solar disk detection | "hough", "edge" | hough usually better |

### Tracking Parameters

| Parameter | Description | Typical Range | Notes |
|-----------|-------------|---------------|--------|
| `max_distance_deg` | Max matching distance | 10-25° | Larger for longer gaps |
| `max_time_days` | Max time gap | 1-10 days | Depends on data cadence |
| `min_track_length` | Min observations per track | 2-5 | Higher for precision |

### Analysis Parameters

| Parameter | Description | Typical Range | Notes |
|-----------|-------------|---------------|--------|
| `fit_differential_rotation` | Enable rotation fit | true/false | Main analysis feature |
| `minimum_tracks_for_fit` | Min tracks for fit | 3-10 | More tracks = better fit |
| `exclude_high_latitude` | Latitude cutoff | 60-75° | Exclude polar regions |

## 📁 Sample Data Organization

Organize your data like this for best results:

```
project/
├── data/
│   ├── 2023-10-15/
│   │   ├── hmi_continuum_001.fits
│   │   ├── hmi_continuum_002.fits
│   │   └── times.txt
│   ├── 2023-11-03/
│   │   ├── hmi_continuum_001.fits
│   │   ├── hmi_continuum_002.fits
│   │   ├── hmi_continuum_003.fits
│   │   └── metadata.json
│   └── configs/
│       ├── high_activity.json
│       └── low_activity.json
├── results/
│   ├── 2023-10-15_analysis/
│   └── 2023-11-03_analysis/
└── scripts/
    ├── batch_process.py
    └── quality_check.py
```

## 🔍 Data Quality Assessment

Before running analysis, check your data quality:

### Image Quality Checklist
- [ ] Full solar disk visible
- [ ] No significant artifacts or cosmic rays
- [ ] Consistent exposure/processing
- [ ] FITS headers contain WCS information
- [ ] Images taken within 1-7 day timespan

### Sunspot Quality Checklist  
- [ ] At least 1-2 sunspots visible
- [ ] Sunspots persist across multiple images
- [ ] Clear contrast against photosphere
- [ ] Not too close to solar limb (< 60° from disk center)

## 🛠️ Troubleshooting Examples

### Problem: No Sunspots Detected
```json
// Try more sensitive settings
{
  "detection": {
    "min_sunspot_area": 3,
    "intensity_threshold": -1.5,
    "disk_method": "edge"
  }
}
```

### Problem: Poor Tracking Performance
```json
// Try more permissive tracking
{
  "tracking": {
    "max_distance_deg": 25.0,
    "max_time_days": 10.0,
    "min_track_length": 2
  }
}
```

### Problem: Fit Quality Too Low
```json
// Require more stringent tracks
{
  "tracking": {
    "min_track_length": 3
  },
  "analysis": {
    "minimum_tracks_for_fit": 5,
    "exclude_high_latitude": 50.0
  }
}
```

## 📈 Performance Optimization

### For Large Datasets
```json
{
  "output": {
    "save_intermediate": false,
    "create_summary_plots": true,
    "plot_individual_detections": false
  },
  "preprocessing": {
    "image_standardization": {
      "target_size": [1024, 1024]
    }
  }
}
```

### For High Precision
```json
{
  "detection": {
    "limb_darkening_correction": true,
    "morphology_operations": {
      "remove_small_objects": true,
      "fill_holes": true
    }
  },
  "preprocessing": {
    "coordinate_system": {
      "use_sunpy_maps": true
    }
  }
}
```

## 🎨 Visualization Options

### Standard Plots
The pipeline automatically creates:
- Solar disk detection overlay
- Sunspot detection masks
- Track evolution plots
- Rotation rate vs latitude
- Differential rotation fit curve

### Custom Plotting
```python
import matplotlib.pyplot as plt
from solar_pipeline import SolarAnalysisPipeline

# After running analysis
results = pipeline.run_tracking_analysis()

# Custom plot
rotation_data = results['rotation_data']
latitudes = [r['mean_latitude'] for r in rotation_data]
rates = [r['rotation_rate_deg_day'] for r in rotation_data]

plt.figure(figsize=(10, 6))
plt.scatter(latitudes, rates, s=100, alpha=0.7)
plt.xlabel('Latitude (degrees)')
plt.ylabel('Rotation Rate (deg/day)')
plt.title('My Custom Solar Rotation Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

## 📊 Result Interpretation

### Excellent Results
- **R² > 0.9**: High confidence in measurements
- **5+ tracks**: Good statistical basis
- **Literature agreement**: Within 2σ of known values
- **Wide latitude range**: > 40° coverage

### Good Results  
- **R² > 0.7**: Reliable measurements
- **3-5 tracks**: Reasonable statistics
- **Moderate agreement**: Some parameters match literature
- **Decent coverage**: > 20° latitude range

### Preliminary Results
- **R² > 0.5**: Trend visible but uncertain
- **2-3 tracks**: Limited statistics
- **Qualitative agreement**: General trend matches expectations
- **Limited coverage**: < 20° latitude range

## 🔗 Integration Examples

### With Jupyter Notebooks
```python
# notebook_analysis.py
%matplotlib inline
import sys
sys.path.append('../')

from solar_pipeline import SolarAnalysisPipeline
import matplotlib.pyplot as plt

# Your analysis here
pipeline = SolarAnalysisPipeline()
# ... rest of analysis

# Results will display inline
```

### With Automated Workflows
```python
# automated_pipeline.py
import schedule
import time
from datetime import datetime, timedelta

def daily_solar_analysis():
    """Run analysis on latest solar data"""
    
    # Download latest data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Run pipeline
    pipeline = SolarAnalysisPipeline()
    # ... analysis code
    
    print(f"Analysis completed at {datetime.now()}")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(daily_solar_analysis)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## 📚 Further Reading

- [Main README](../README.md) - Complete pipeline documentation
- [Installation Guide](../docs/installation.md) - Detailed setup instructions  
- [API Reference](../docs/api_reference.md) - Complete function documentation
- [Scientific Background](../docs/scientific_background.md) - Theory and methods

## 💡 Tips for Success

1. **Start Small**: Begin with 3-4 high-quality images
2. **Check Results**: Always examine the generated plots
3. **Iterate**: Adjust parameters based on initial results
4. **Document**: Keep track of what settings work for your data
5. **Validate**: Compare results with literature when possible

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Examine the `analysis_report.txt` file generated
3. Look at intermediate results in `solar_analysis_results/`
4. Enable debug logging in the pipeline
5. Try different parameter combinations from the examples

---

**Happy analyzing!** 🌞
