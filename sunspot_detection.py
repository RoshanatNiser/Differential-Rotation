import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import sunpy.map
from sunpy.coordinates import frames, get_earth
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import cv2
from scipy import ndimage
from skimage import measure, morphology, filters, exposure
from skimage.feature import canny, peak_local_max
import warnings
warnings.filterwarnings('ignore')


class SolarDiskDetector:
    """Class to detect solar disk and extract sunspots"""
    
    def __init__(self):
        self.solar_radius_pixels = None
        self.solar_center = None
        
    def load_image(self, filepath):
        """Load FITS or JPG image"""
        if filepath.lower().endswith('.fits'):
            with fits.open(filepath) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
                return image_data, header
        else:
            # For JPG/PNG files - load in color first to handle RGB properly
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {filepath}")
            # Convert to grayscale properly
            image_data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            return image_data, None
    
    def detect_solar_disk(self, image_data, method='hough'):
        """
        Detect solar disk using circular Hough transform or edge detection
        """
        
        if method == 'hough':
            return self._detect_disk_hough(image_data)
        else:
            return self._detect_disk_edge(image_data)
    
    def _detect_disk_hough(self, image_data):
        """Detect solar disk using Hough circle transform"""
        
        # Normalize and convert to uint8
        img_normalized = ((image_data - np.min(image_data)) / 
                         (np.max(image_data) - np.min(image_data)) * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(img_normalized, (15, 15), 0)
        
        # More aggressive Hough parameters for better detection
        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(image_data.shape) * 0.5),
            param1=30,
            param2=20,
            minRadius=int(min(image_data.shape) * 0.25),
            maxRadius=int(min(image_data.shape) * 0.7)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            center_x, center_y, radius = circles[0]
            center = (center_x, center_y)
            
            # Create mask
            mask = np.zeros(image_data.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 1, -1)
            
            self.solar_center = center
            self.solar_radius_pixels = radius
            
            return center, radius, mask.astype(bool)
        else:
            raise ValueError("Could not detect solar disk using Hough transform")
    
    def _detect_disk_edge(self, image_data):
        """Detect solar disk using edge detection and fitting"""
        
        # Normalize image
        img_norm = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        
        # Apply Gaussian filter
        smoothed = filters.gaussian(img_norm, sigma=3)
        
        # Use Sobel edge detection
        edges = filters.sobel(smoothed)
        
        # Threshold edges
        edge_threshold = np.percentile(edges, 90)
        edges_binary = edges > edge_threshold
        
        # Find contours
        contours = measure.find_contours(edges_binary, 0.5)
        
        if not contours:
            raise ValueError("No contours found for solar disk detection")
        
        # Find the longest contour
        longest_contour = max(contours, key=len)
        
        # Fit a circle to the contour points
        center, radius = self._fit_circle(longest_contour)
        
        # Create mask
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        self.solar_center = center
        self.solar_radius_pixels = radius
        
        return center, radius, mask
    
    def _fit_circle(self, points):
        """Fit a circle to a set of points using least squares"""
        x = points[:, 1]
        y = points[:, 0]
        
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        center_x = coeffs[0]
        center_y = coeffs[1]
        radius = np.sqrt(coeffs[2] + center_x**2 + center_y**2)
        
        return (center_x, center_y), radius


class SunspotDetector:
    """Class to detect and analyze sunspots"""
    
    def __init__(self, solar_center, solar_radius):
        self.solar_center = solar_center
        self.solar_radius = solar_radius
    
    def detect_sunspots(self, image_data, solar_mask, min_area=5, intensity_threshold=None):
        """
        Detect sunspots in the solar disk with enhanced sensitivity
        """
        
        # Apply solar disk mask
        masked_image = image_data.copy().astype(float)
        masked_image[~solar_mask] = np.nan
        
        # Get statistics before correction
        valid_original = image_data[solar_mask]
        print(f"Original image stats: mean={np.mean(valid_original):.2f}, std={np.std(valid_original):.2f}")
        print(f"Original range: [{np.min(valid_original):.2f}, {np.max(valid_original):.2f}]")
        
        # Apply contrast enhancement on the disk region
        disk_only = image_data.copy().astype(float)
        disk_only[~solar_mask] = 0
        
        # Adaptive histogram equalization within the solar disk
        disk_uint8 = ((disk_only - disk_only[solar_mask].min()) / 
                     (disk_only[solar_mask].max() - disk_only[solar_mask].min()) * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(disk_uint8)
        enhanced = enhanced.astype(float)
        enhanced[~solar_mask] = np.nan
        
        print(f"Enhanced image stats: mean={np.nanmean(enhanced):.2f}, std={np.nanstd(enhanced):.2f}")
        
        # Apply limb darkening correction
        corrected_image = self._correct_limb_darkening(masked_image, solar_mask)
        
        # Use multiple detection strategies
        sunspot_masks = []
        
        # Strategy 1: Enhanced image with aggressive threshold
        valid_enhanced = enhanced[solar_mask & ~np.isnan(enhanced)]
        if len(valid_enhanced) > 0:
            threshold1 = np.nanmean(valid_enhanced) - 1.0 * np.nanstd(valid_enhanced)
            mask1 = (enhanced < threshold1) & solar_mask & ~np.isnan(enhanced)
            sunspot_masks.append(mask1)
            print(f"Strategy 1 threshold: {threshold1:.2f}, found {np.sum(mask1)} pixels")
        
        # Strategy 2: Limb-corrected image
        valid_corrected = corrected_image[solar_mask & ~np.isnan(corrected_image)]
        if len(valid_corrected) > 0:
            threshold2 = np.nanmean(valid_corrected) - 1.2 * np.nanstd(valid_corrected)
            mask2 = (corrected_image < threshold2) & solar_mask & ~np.isnan(corrected_image)
            sunspot_masks.append(mask2)
            print(f"Strategy 2 threshold: {threshold2:.2f}, found {np.sum(mask2)} pixels")
        
        # Strategy 3: Original image with percentile-based threshold
        threshold3 = np.percentile(valid_original, 15)  # Darkest 15%
        mask3 = (image_data < threshold3) & solar_mask
        sunspot_masks.append(mask3)
        print(f"Strategy 3 threshold (15th percentile): {threshold3:.2f}, found {np.sum(mask3)} pixels")
        
        # Strategy 4: Local minima detection
        # Find local minima in the solar disk
        local_min = filters.rank.minimum(disk_uint8, morphology.disk(3))
        local_min = local_min.astype(float)
        local_min[~solar_mask] = 255
        threshold4 = np.percentile(local_min[solar_mask], 10)
        mask4 = (local_min < threshold4) & solar_mask
        sunspot_masks.append(mask4)
        print(f"Strategy 4 (local minima): found {np.sum(mask4)} pixels")
        
        # Combine all strategies
        combined_mask = np.zeros_like(image_data, dtype=bool)
        for mask in sunspot_masks:
            combined_mask = combined_mask | mask
        
        print(f"Combined mask: {np.sum(combined_mask)} pixels")
        
        # Clean up the mask
        combined_mask = morphology.binary_opening(combined_mask, morphology.disk(1))
        combined_mask = morphology.remove_small_objects(combined_mask, min_size=min_area)
        combined_mask = morphology.remove_small_holes(combined_mask, area_threshold=min_area*2)
        
        print(f"After cleanup: {np.sum(combined_mask)} pixels")
        
        # Label connected components
        labeled_spots = measure.label(combined_mask)
        num_regions = labeled_spots.max()
        print(f"Found {num_regions} connected regions")
        
        # Extract sunspot properties
        sunspots = []
        for region in measure.regionprops(labeled_spots, intensity_image=image_data):
            if region.area >= min_area:
                centroid_x = region.centroid[1]
                centroid_y = region.centroid[0]
                
                sunspot = {
                    'centroid': (centroid_x, centroid_y),
                    'centroid_rowcol': region.centroid,
                    'area': region.area,
                    'mean_intensity': region.mean_intensity,
                    'min_intensity': region.min_intensity,
                    'bbox': region.bbox,
                    'equivalent_diameter': region.equivalent_diameter,
                    'eccentricity': region.eccentricity
                }
                sunspots.append(sunspot)
        
        return sunspots, combined_mask
    
    def _correct_limb_darkening(self, image_data, solar_mask):
        """Apply limb darkening correction"""
        
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        distance_from_center = np.sqrt((x - self.solar_center[0])**2 + 
                                     (y - self.solar_center[1])**2)
        
        normalized_distance = distance_from_center / self.solar_radius
        normalized_distance = np.clip(normalized_distance, 0, 0.99)
        
        mu_squared = 1 - normalized_distance**2
        mu_squared = np.maximum(mu_squared, 1e-10)
        mu = np.sqrt(mu_squared)
        
        u = 0.6
        limb_darkening_factor = 1 - u + u * mu
        limb_darkening_factor = np.maximum(limb_darkening_factor, 0.1)
        
        corrected_image = image_data / limb_darkening_factor
        corrected_image[~solar_mask] = np.nan
        
        return corrected_image
    
    def pixel_to_heliographic(self, pixel_coords, observation_time, 
                            plate_scale=None, use_ephemeris=True):
        """Convert pixel coordinates to heliographic coordinates"""
        
        if isinstance(observation_time, str):
            obs_time = Time(observation_time)
        else:
            obs_time = observation_time
        
        solar_radius_arcsec = 959.63
        
        if plate_scale is None:
            plate_scale = solar_radius_arcsec / self.solar_radius
        
        if isinstance(pixel_coords[0], (int, float)):
            pixel_coords = [pixel_coords]
        
        try:
            earth_coord = get_earth(obs_time)
        except Exception as e:
            print(f"Warning: Could not get Earth coordinates: {e}")
            earth_coord = 'earth'
        
        helio_coords = []
        for x_pix, y_pix in pixel_coords:
            try:
                dx_arcsec = (x_pix - self.solar_center[0]) * plate_scale
                dy_arcsec = (y_pix - self.solar_center[1]) * plate_scale
                
                helio_coord = SkyCoord(
                    dx_arcsec * u.arcsec,
                    dy_arcsec * u.arcsec,
                    frame=frames.Helioprojective,
                    obstime=obs_time,
                    observer=earth_coord
                ).transform_to(frames.HeliographicStonyhurst)
                
                helio_coords.append(helio_coord)
                
            except Exception as e:
                print(f"Error converting coordinates for pixel ({x_pix}, {y_pix}): {e}")
                helio_coord = SkyCoord(
                    0 * u.deg, 0 * u.deg,
                    frame=frames.HeliographicStonyhurst,
                    obstime=obs_time
                )
                helio_coords.append(helio_coord)
        
        return helio_coords


def main_analysis(image_path, observation_time, min_area=5, intensity_threshold=None, 
                 disk_method='hough', use_ephemeris=True, **kwargs):
    """
    Main function to perform complete sunspot analysis
    """
    
    disk_detector = SolarDiskDetector()
    
    print("Loading image...")
    image_data, header = disk_detector.load_image(image_path)
    print(f"Image shape: {image_data.shape}, dtype: {image_data.dtype}")
    print(f"Image range: [{np.min(image_data)}, {np.max(image_data)}]")
    
    print("\nDetecting solar disk...")
    try:
        center, radius, solar_mask = disk_detector.detect_solar_disk(image_data, method=disk_method)
        print(f"Solar disk detected: center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f} pixels")
    except Exception as e:
        print(f"Error with {disk_method} method: {e}")
        print("Trying alternate method...")
        alt_method = 'edge' if disk_method == 'hough' else 'hough'
        center, radius, solar_mask = disk_detector.detect_solar_disk(image_data, method=alt_method)
        print(f"Solar disk detected: center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f} pixels")
    
    sunspot_detector = SunspotDetector(center, radius)
    
    print("\nDetecting sunspots...")
    sunspots, sunspot_mask = sunspot_detector.detect_sunspots(
        image_data, solar_mask, min_area=min_area, intensity_threshold=intensity_threshold
    )
    print(f"\n*** Found {len(sunspots)} sunspots ***\n")
    
    if len(sunspots) > 0:
        print("Converting to heliographic coordinates...")
        for i, sunspot in enumerate(sunspots):
            try:
                helio_coords = sunspot_detector.pixel_to_heliographic(
                    sunspot['centroid'],
                    observation_time,
                    use_ephemeris=use_ephemeris
                )
                if helio_coords:
                    sunspot['heliographic_lon'] = helio_coords[0].lon
                    sunspot['heliographic_lat'] = helio_coords[0].lat
                    print(f"Sunspot {i+1}: Lon={helio_coords[0].lon.deg:.2f}°, "
                          f"Lat={helio_coords[0].lat.deg:.2f}°, Area={sunspot['area']} px")
            except Exception as e:
                print(f"Could not convert sunspot {i+1} coordinates: {e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Original image with enhanced contrast for visualization
    vmin, vmax = np.percentile(image_data[solar_mask], [1, 99])
    axes[0,0].imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0,0].set_title(f'Original Solar Image\n(contrast enhanced for display)', fontsize=12, fontweight='bold')
    circle = plt.Circle(center, radius, fill=False, color='red', linewidth=2)
    axes[0,0].add_patch(circle)
    axes[0,0].set_xlabel('X (pixels)')
    axes[0,0].set_ylabel('Y (pixels)')
    
    # Solar disk mask
    axes[0,1].imshow(solar_mask, cmap='gray', origin='lower')
    axes[0,1].set_title('Solar Disk Mask', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('X (pixels)')
    axes[0,1].set_ylabel('Y (pixels)')
    
    # Sunspot mask
    axes[1,0].imshow(sunspot_mask, cmap='hot', origin='lower')
    axes[1,0].set_title(f'Detected Sunspots: {len(sunspots)} regions', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('X (pixels)')
    axes[1,0].set_ylabel('Y (pixels)')
    
    # Overlay
    axes[1,1].imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1,1].imshow(sunspot_mask, cmap='Reds', alpha=0.6, origin='lower')
    axes[1,1].set_title('Sunspots Overlay', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('X (pixels)')
    axes[1,1].set_ylabel('Y (pixels)')
    
    # Mark sunspot centroids
    for i, sunspot in enumerate(sunspots):
        x, y = sunspot['centroid']
        if not (np.isnan(x) or np.isnan(y)):
            axes[1,1].plot(x, y, 'r+', markersize=15, markeredgewidth=3)
            axes[1,1].text(x+20, y+20, f'{i+1}', color='yellow', 
                          fontweight='bold', fontsize=14, 
                          bbox=dict(boxstyle='round', facecolor='red', alpha=0.8, edgecolor='white', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('sunspot_detection_result.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'sunspot_detection_result.png'")
    plt.show()
    
    return sunspots, {'center': center, 'radius': radius, 'mask': solar_mask}


if __name__ == "__main__":
    image_path = "solar_image.fits"
    observation_time = "2023-08-15T12:00:00"
    
    try:
        sunspots, solar_info = main_analysis(
            image_path, 
            observation_time, 
            min_area=3,  # Very low threshold
            disk_method='hough',
            use_ephemeris=True
        )
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Solar disk center: ({solar_info['center'][0]:.1f}, {solar_info['center'][1]:.1f})")
        print(f"Solar disk radius: {solar_info['radius']:.1f} pixels")
        print(f"Number of sunspots detected: {len(sunspots)}")
        
        if len(sunspots) > 0:
            print("\nDetailed Sunspot Information:")
            print("-"*60)
            for i, spot in enumerate(sunspots):
                print(f"\nSunspot {i+1}:")
                print(f"  Position (x, y): ({spot['centroid'][0]:.1f}, {spot['centroid'][1]:.1f})")
                print(f"  Area: {spot['area']} pixels")
                print(f"  Equivalent diameter: {spot['equivalent_diameter']:.1f} pixels")
                print(f"  Mean intensity: {spot['mean_intensity']:.2f}")
                if 'heliographic_lon' in spot:
                    print(f"  Heliographic longitude: {spot['heliographic_lon'].deg:.2f}°")
                    print(f"  Heliographic latitude: {spot['heliographic_lat'].deg:.2f}°")
        else:
            print("\nNo sunspots detected. Try:")
            print("  - Reducing min_area parameter")
            print("  - Using a different image with more visible sunspots")
            print("  - Checking if the image quality is sufficient")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()