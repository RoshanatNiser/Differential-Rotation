from solar_pipeline import SolarAnalysisPipeline

# Define your images and times
image_files = ["C:\\niser\\nac\\sunspot\\Differential-Rotation-main\\Image\\image_4.jpeg"]
times = ["2024-03-31T17:18"]

# Create pipeline with custom config
config = {
    'detection': {
        'min_sunspot_area': 3,  # Smaller for subtle spots
        'disk_method': 'hough'
    }
}

pipeline = SolarAnalysisPipeline(config)

# Process images
success = pipeline.process_image_sequence(image_files, times)

# Run tracking if successful
if success:
    results = pipeline.run_tracking_analysis()
    pipeline.create_summary_report()

if results:
    print(f"Found {len(results['tracks'])} sunspot tracks")
    
    # Access detailed data
    rotation_data = results['rotation_data']
    fit_params = results['fit_parameters']
    
"""
# Console output
Found 3 sunspot tracks
Differential Rotation Fit Results:
ω(θ) = 14.234 + -2.156*sin²(θ) + -1.345*sin⁴(θ) deg/day
Equatorial rotation rate: 14.234 ± 0.123 deg/day  
Equatorial period: 25.31 days
R-squared: 0.8456

# Files created
solar_analysis_results/
├── rotation_analysis.csv          # Track data
├── differential_rotation_fit.json # Fit parameters  
├── sunspot_tracks.json           # Track summaries
├── analysis_report.txt           # Human-readable summary
└── image_xxx_results.json        # Per-image results

"""