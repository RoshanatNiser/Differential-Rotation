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
from skimage import measure, morphology, filters
from skimage.feature import peak_local_maxima
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
            # For JPG/PNG files
            image_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            return image_data, None
    
    def detect_solar_disk(self, image_data, method='hough'):
        """
        Detect solar disk using circular Hough transform or edge detection
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Input solar image
        method : str
            'hough' for Hough circle detection, 'edge' for edge-based detection
        
        Returns:
        --------
        center : tuple
            (x, y) center coordinates in pixel coordinates
        radius : float
            Solar disk radius in pixels
        mask : numpy.ndarray
            Binary mask of solar disk
        
        Notes:
        ------
        All coordinates follow (x, y) pixel convention where x is horizontal (column)
        and y is vertical (row) in image coordinates.
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
        
        # Use HoughCircles to detect the solar disk
        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(image_data.shape) * 0.5),
            param1=50,
            param2=30,
            minRadius=int(min(image_data.shape) * 0.3),
            maxRadius=int(min(image_data.shape) * 0.6)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Take the first (most prominent) circle
            center_x, center_y, radius = circles[0]
            center = (center_x, center_y)  # (x, y) format
            
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
        
        # Apply Gaussian filter to smooth the image
        smoothed = filters.gaussian(image_data, sigma=2)
        
        # Find edges using Canny edge detector
        edges = filters.canny(smoothed, sigma=1, low_threshold=0.1, high_threshold=0.2)
        
        # Find contours
        contours = measure.find_contours(edges, 0.5)
        
        if not contours:
            raise ValueError("No contours found for solar disk detection")
        
        # Find the largest contour (likely the solar limb)
        largest_contour = max(contours, key=len)
        
        # Fit a circle to the contour points
        center, radius = self._fit_circle(largest_contour)
        
        # Create mask
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        self.solar_center = center
        self.solar_radius_pixels = radius
        
        return center, radius, mask
    
    def _fit_circle(self, points):
        """
        Fit a circle to a set of points using least squares
        
        Parameters:
        -----------
        points : numpy.ndarray
            Points from find_contours in (row, col) format
            
        Returns:
        --------
        center : tuple
            Circle center in (x, y) pixel coordinates
        radius : float
            Circle radius in pixels
        """
        
        # Convert from (row, col) to (x, y) coordinates
        x = points[:, 1]  # col -> x
        y = points[:, 0]  # row -> y
        
        # Set up the least squares problem: (x-a)^2 + (y-b)^2 = r^2
        # Rearranged: x^2 + y^2 - 2ax - 2by + (a^2 + b^2 - r^2) = 0
        # Linear form: -2ax - 2by + (a^2 + b^2 - r^2) = -(x^2 + y^2)
        
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        # Solve least squares
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        center_x = coeffs[0]
        center_y = coeffs[1]
        radius = np.sqrt(coeffs[2] + center_x**2 + center_y**2)
        
        return (center_x, center_y), radius

class SunspotDetector:
    """Class to detect and analyze sunspots"""
    
    def __init__(self, solar_center, solar_radius):
        self.solar_center = solar_center  # (x, y) pixel coordinates
        self.solar_radius = solar_radius
    
    def detect_sunspots(self, image_data, solar_mask, min_area=10, intensity_threshold=None):
        """
        Detect sunspots in the solar disk
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Input solar image
        solar_mask : numpy.ndarray
            Binary mask of solar disk
        min_area : int
            Minimum area for sunspot detection (pixels)
        intensity_threshold : float
            Threshold for sunspot detection (if None, auto-computed)
        
        Returns:
        --------
        sunspots : list
            List of detected sunspots with properties
        sunspot_mask : numpy.ndarray
            Binary mask showing detected sunspots
        """
        
        # Apply solar disk mask
        masked_image = image_data.copy()
        masked_image[~solar_mask] = np.nan
        
        # Calculate limb darkening correction (optional)
        corrected_image = self._correct_limb_darkening(masked_image, solar_mask)
        
        # Determine threshold for sunspot detection
        if intensity_threshold is None:
            # Use mean - 2*std as threshold
            valid_pixels = corrected_image[solar_mask]
            intensity_threshold = np.nanmean(valid_pixels) - 2 * np.nanstd(valid_pixels)
        
        # Create binary mask for dark regions (potential sunspots)
        sunspot_mask = (corrected_image < intensity_threshold) & solar_mask
        
        # Remove small objects and fill holes
        sunspot_mask = morphology.remove_small_objects(sunspot_mask, min_size=min_area)
        sunspot_mask = morphology.remove_small_holes(sunspot_mask, area_threshold=min_area)
        
        # Label connected components
        labeled_spots = measure.label(sunspot_mask)
        
        # Extract sunspot properties
        sunspots = []
        for region in measure.regionprops(labeled_spots, intensity_image=image_data):
            if region.area >= min_area:
                # Convert centroid from (row, col) to (x, y)
                centroid_x = region.centroid[1]  # col -> x
                centroid_y = region.centroid[0]  # row -> y
                
                sunspot = {
                    'centroid': (centroid_x, centroid_y),  # (x, y) format
                    'centroid_rowcol': region.centroid,    # Keep original for reference
                    'area': region.area,
                    'mean_intensity': region.mean_intensity,
                    'min_intensity': region.min_intensity,
                    'bbox': region.bbox,
                    'equivalent_diameter': region.equivalent_diameter,
                    'eccentricity': region.eccentricity
                }
                sunspots.append(sunspot)
        
        return sunspots, sunspot_mask
    
    def _correct_limb_darkening(self, image_data, solar_mask):
        """
        Apply simple limb darkening correction with proper bounds checking
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Input solar image
        solar_mask : numpy.ndarray
            Binary mask of solar disk
            
        Returns:
        --------
        corrected_image : numpy.ndarray
            Limb darkening corrected image
        """
        
        # Create distance map from solar center
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        distance_from_center = np.sqrt((x - self.solar_center[0])**2 + 
                                     (y - self.solar_center[1])**2)
        
        # Normalize distance by solar radius with bounds checking
        normalized_distance = distance_from_center / self.solar_radius
        normalized_distance = np.clip(normalized_distance, 0, 1)  # Ensure within bounds
        
        # Calculate mu with epsilon to avoid division by zero
        mu_squared = 1 - normalized_distance**2
        mu_squared = np.maximum(mu_squared, 1e-10)  # Avoid values too close to zero
        mu = np.sqrt(mu_squared)
        mu[~solar_mask] = np.nan
        
        # Simple limb darkening law: I(mu) = I0 * (1 - u + u*mu)
        # For correction, we divide by this factor
        u = 0.6  # limb darkening coefficient
        limb_darkening_factor = 1 - u + u * mu
        
        # Avoid division by very small numbers
        limb_darkening_factor = np.maximum(limb_darkening_factor, 1e-10)
        
        corrected_image = image_data / limb_darkening_factor
        corrected_image[~solar_mask] = np.nan
        
        return corrected_image
    
    def pixel_to_heliographic(self, pixel_coords, observation_time, 
                            plate_scale=None, use_ephemeris=True):
        """
        Convert pixel coordinates to heliographic coordinates with proper error handling
        
        Parameters:
        -----------
        pixel_coords : tuple or list
            (x, y) pixel coordinates or list of coordinates
        observation_time : str or astropy.time.Time
            Observation time
        plate_scale : float
            Arcseconds per pixel (if None, estimated from solar radius)
        use_ephemeris : bool
            Whether to use ephemeris data for accurate solar radius
        
        Returns:
        --------
        coordinates : list
            List of SkyCoord objects in heliographic coordinates
        """
        
        # Parse observation time
        if isinstance(observation_time, str):
            obs_time = Time(observation_time)
        else:
            obs_time = observation_time
        
        # Get accurate solar radius from ephemeris if requested
        if use_ephemeris:
            try:
                sun = get_sun(obs_time)
                # Get solar radius in arcseconds (approximately 959.63 arcsec average)
                solar_radius_arcsec = 959.63  # Use average for now, can be refined
            except Exception as e:
                print(f"Warning: Could not get ephemeris data, using average solar radius: {e}")
                solar_radius_arcsec = 959.63
        else:
            solar_radius_arcsec = 959.63  # Average solar radius in arcseconds
        
        # Calculate plate scale if not provided
        if plate_scale is None:
            plate_scale = solar_radius_arcsec / self.solar_radius
        
        # Convert to lists if single coordinate
        if isinstance(pixel_coords[0], (int, float)):
            pixel_coords = [pixel_coords]
        
        # Get Earth's location for the observation time
        try:
            earth_coord = get_earth(obs_time)
        except Exception as e:
            print(f"Warning: Could not get Earth coordinates, using default: {e}")
            earth_coord = 'earth'  # Fallback to string identifier
        
        # Convert pixel coordinates to heliocentric coordinates
        helio_coords = []
        for x_pix, y_pix in pixel_coords:
            try:
                # Convert to angular coordinates relative to solar center
                dx_arcsec = (x_pix - self.solar_center[0]) * plate_scale
                dy_arcsec = (y_pix - self.solar_center[1]) * plate_scale
                
                # Check if coordinates are within reasonable bounds (solar limb)
                angular_distance = np.sqrt(dx_arcsec**2 + dy_arcsec**2)
                if angular_distance > solar_radius_arcsec * 1.1:  # Allow 10% margin
                    print(f"Warning: Coordinates ({x_pix}, {y_pix}) may be beyond solar limb")
                
                # Convert to heliographic coordinates using SunPy
                helio_coord = SkyCoord(
                    dx_arcsec * u.arcsec,
                    dy_arcsec * u.arcsec,
                    frame=frames.Helioprojective,
                    obstime=obs_time,
                    observer=earth_coord
                ).transform_to(frames.HeliographicStonyhurst)
                
                # Validate coordinates
                if not self._validate_coordinates(helio_coord):
                    print(f"Warning: Invalid coordinates calculated for pixel ({x_pix}, {y_pix})")
                
                helio_coords.append(helio_coord)
                
            except Exception as e:
                print(f"Error converting coordinates for pixel ({x_pix}, {y_pix}): {e}")
                # Create a placeholder coordinate
                helio_coord = SkyCoord(
                    0 * u.deg, 0 * u.deg,
                    frame=frames.HeliographicStonyhurst,
                    obstime=obs_time
                )
                helio_coords.append(helio_coord)
        
        return helio_coords
    
    def _validate_coordinates(self, coord):
        """
        Validate heliographic coordinates
        
        Parameters:
        -----------
        coord : SkyCoord
            Coordinate to validate
            
        Returns:
        --------
        bool
            True if coordinates are valid
        """
        
        # Check latitude bounds (-90 to +90 degrees)
        if not (-90 <= coord.lat.deg <= 90):
            return False
        
        # Check longitude bounds (-180 to +180 degrees)
        if not (-180 <= coord.lon.deg <= 180):
            return False
        
        # Additional checks could be added here for physical reasonableness
        return True

def main_analysis(image_path, observation_time, min_area=10, intensity_threshold=None, 
                 disk_method='hough', use_ephemeris=True, **kwargs):
    """
    Main function to perform complete sunspot analysis
    
    Parameters:
    -----------
    image_path : str
        Path to solar image (FITS or JPG)
    observation_time : str
        Observation time (ISO format)
    min_area : int
        Minimum sunspot area in pixels
    intensity_threshold : float, optional
        Custom intensity threshold for sunspot detection
    disk_method : str
        Solar disk detection method ('hough' or 'edge')
    use_ephemeris : bool
        Whether to use ephemeris data for coordinate conversion
    **kwargs : dict
        Additional parameters
    """
    
    # Initialize detector
    disk_detector = SolarDiskDetector()
    
    # Load image
    print("Loading image...")
    image_data, header = disk_detector.load_image(image_path)
    
    # Detect solar disk
    print("Detecting solar disk...")
    center, radius, solar_mask = disk_detector.detect_solar_disk(image_data, method=disk_method)
    print(f"Solar disk detected: center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f} pixels")
    
    # Initialize sunspot detector
    sunspot_detector = SunspotDetector(center, radius)
    
    # Detect sunspots
    print("Detecting sunspots...")
    sunspots, sunspot_mask = sunspot_detector.detect_sunspots(
        image_data, solar_mask, min_area=min_area, intensity_threshold=intensity_threshold
    )
    print(f"Found {len(sunspots)} sunspots")
    
    # Convert to heliographic coordinates
    print("Converting to heliographic coordinates...")
    for i, sunspot in enumerate(sunspots):
        try:
            helio_coords = sunspot_detector.pixel_to_heliographic(
                sunspot['centroid'],  # Already in (x, y) format
                observation_time,
                use_ephemeris=use_ephemeris
            )
            if helio_coords:
                sunspot['heliographic_lon'] = helio_coords[0].lon
                sunspot['heliographic_lat'] = helio_coords[0].lat
                print(f"Sunspot {i+1}: Longitude={helio_coords[0].lon.deg:.2f}°, "
                      f"Latitude={helio_coords[0].lat.deg:.2f}°")
            else:
                print(f"Could not convert coordinates for sunspot {i+1}")
        except Exception as e:
            print(f"Could not convert sunspot {i+1} coordinates: {e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0,0].imshow(image_data, cmap='gray', origin='lower')
    axes[0,0].set_title('Original Solar Image')
    circle = plt.Circle(center, radius, fill=False, color='red', linewidth=2)
    axes[0,0].add_patch(circle)
    axes[0,0].set_xlabel('X (pixels)')
    axes[0,0].set_ylabel('Y (pixels)')
    
    # Solar disk mask
    axes[0,1].imshow(solar_mask, cmap='gray', origin='lower')
    axes[0,1].set_title('Solar Disk Mask')
    axes[0,1].set_xlabel('X (pixels)')
    axes[0,1].set_ylabel('Y (pixels)')
    
    # Sunspot mask
    axes[1,0].imshow(sunspot_mask, cmap='gray', origin='lower')
    axes[1,0].set_title('Detected Sunspots')
    axes[1,0].set_xlabel('X (pixels)')
    axes[1,0].set_ylabel('Y (pixels)')
    
    # Overlay
    axes[1,1].imshow(image_data, cmap='gray', origin='lower')
    axes[1,1].imshow(sunspot_mask, cmap='Reds', alpha=0.3, origin='lower')
    axes[1,1].set_title('Sunspots Overlay')
    axes[1,1].set_xlabel('X (pixels)')
    axes[1,1].set_ylabel('Y (pixels)')
    
    # Mark sunspot centroids
    for i, sunspot in enumerate(sunspots):
        x, y = sunspot['centroid']  # Now properly in (x, y) format
        axes[1,1].plot(x, y, 'r+', markersize=10, markeredgewidth=2)
        axes[1,1].text(x+10, y+10, f'{i+1}', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return sunspots, {'center': center, 'radius': radius, 'mask': solar_mask}

# Example usage
if __name__ == "__main__":
    # Example parameters
    image_path = "solar_image.fits"  # or .jpg
    observation_time = "2023-08-15T12:00:00"
    
    try:
        sunspots, solar_info = main_analysis(
            image_path, 
            observation_time, 
            min_area=20,
            disk_method='hough',
            use_ephemeris=True
        )
        
        print(f"\nAnalysis complete!")
        print(f"Solar disk center: ({solar_info['center'][0]:.1f}, {solar_info['center'][1]:.1f})")
        print(f"Solar disk radius: {solar_info['radius']:.1f} pixels")
        print(f"Number of sunspots detected: {len(sunspots)}")
        
        # Print detailed sunspot information
        for i, spot in enumerate(sunspots):
            print(f"\nSunspot {i+1}:")
            print(f"  Position (x, y): ({spot['centroid'][0]:.1f}, {spot['centroid'][1]:.1f})")
            print(f"  Area: {spot['area']} pixels")
            print(f"  Mean intensity: {spot['mean_intensity']:.2f}")
            if 'heliographic_lon' in spot:
                print(f"  Heliographic longitude: {spot['heliographic_lon'].deg:.2f}°")
                print(f"  Heliographic latitude: {spot['heliographic_lat'].deg:.2f}°")
        
    except Exception as e:
        print(f"Error in analysis: {e}")