#sunspot_tracking


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import diff_rot
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class SunspotTracker:
    """Class to track sunspots across multiple observations"""
    
    def __init__(self):
        self.observations = []
        self.sunspot_tracks = []
        
    def add_observation(self, sunspots, observation_time, image_info):
        """
        Add a new observation to the tracking system
        
        Parameters:
        -----------
        sunspots : list
            List of detected sunspots with heliographic coordinates
        observation_time : str or Time
            Observation time
        image_info : dict
            Information about the solar disk detection
        """
        
        observation = {
            'time': Time(observation_time) if isinstance(observation_time, str) else observation_time,
            'sunspots': sunspots,
            'image_info': image_info
        }
        
        self.observations.append(observation)
        print(f"Added observation at {observation['time'].iso} with {len(sunspots)} sunspots")
    
    def match_sunspots(self, max_distance_deg=15.0, max_time_days=7.0):
        """
        Match sunspots across observations to create tracks
        
        Parameters:
        -----------
        max_distance_deg : float
            Maximum distance in degrees to consider sunspots as same
        max_time_days : float
            Maximum time difference in days for matching
        """
        
        if len(self.observations) < 2:
            print("Need at least 2 observations for tracking")
            return
        
        # Sort observations by time
        self.observations.sort(key=lambda x: x['time'])
        
        # Initialize tracks with first observation
        self.sunspot_tracks = []
        for i, sunspot in enumerate(self.observations[0]['sunspots']):
            if 'heliographic_lon' in sunspot and 'heliographic_lat' in sunspot:
                track = {
                    'track_id': i,
                    'observations': [(self.observations[0]['time'], sunspot)],
                    'active': True
                }
                self.sunspot_tracks.append(track)
        
        # Match sunspots in subsequent observations
        for obs_idx in range(1, len(self.observations)):
            current_obs = self.observations[obs_idx]
            current_sunspots = [s for s in current_obs['sunspots'] 
                              if 'heliographic_lon' in s and 'heliographic_lat' in s]
            
            if not current_sunspots:
                continue
            
            # Find active tracks that could be matched
            active_tracks = [t for t in self.sunspot_tracks if t['active']]
            
            if not active_tracks:
                # Create new tracks for all current sunspots
                for i, sunspot in enumerate(current_sunspots):
                    track = {
                        'track_id': len(self.sunspot_tracks),
                        'observations': [(current_obs['time'], sunspot)],
                        'active': True
                    }
                    self.sunspot_tracks.append(track)
                continue
            
            # Get coordinates of last observation for each active track
            track_coords = []
            track_times = []
            for track in active_tracks:
                last_time, last_spot = track['observations'][-1]
                track_coords.append([
                    float(last_spot['heliographic_lon'].to_value(u.deg)),
                    float(last_spot['heliographic_lat'].to_value(u.deg))
                ])
                track_times.append(last_time)
            
            # Get coordinates of current sunspots
            current_coords = []
            for spot in current_sunspots:
                current_coords.append([
                    float(spot['heliographic_lon'].to_value(u.deg)),
                    float(spot['heliographic_lat'].to_value(u.deg))
                ])
            
            if not track_coords or not current_coords:
                continue
            
            track_coords = np.array(track_coords)
            current_coords = np.array(current_coords)
            
            # Calculate distances between tracks and current sunspots
            distances = cdist(track_coords, current_coords, metric='euclidean')
            
            # Apply time constraint
            time_diffs = [(current_obs['time'] - t).to_value(u.day) for t in track_times]
            for i, time_diff in enumerate(time_diffs):
                if time_diff > max_time_days:
                    distances[i, :] = np.inf
            
            # Match using minimum distance with threshold
            matched_tracks = set()
            matched_spots = set()
            
            for _ in range(min(len(active_tracks), len(current_sunspots))):
                min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
                track_idx, spot_idx = min_dist_idx
                min_dist = distances[track_idx, spot_idx]
                
                if min_dist <= max_distance_deg:
                    # Match found
                    active_tracks[track_idx]['observations'].append(
                        (current_obs['time'], current_sunspots[spot_idx])
                    )
                    matched_tracks.add(track_idx)
                    matched_spots.add(spot_idx)
                    
                    # Remove this pair from further consideration
                    distances[track_idx, :] = np.inf
                    distances[:, spot_idx] = np.inf
                else:
                    break
            
            # Mark unmatched tracks as inactive
            for i, track in enumerate(active_tracks):
                if i not in matched_tracks:
                    track['active'] = False
            
            # Create new tracks for unmatched sunspots
            for i, sunspot in enumerate(current_sunspots):
                if i not in matched_spots:
                    track = {
                        'track_id': len(self.sunspot_tracks),
                        'observations': [(current_obs['time'], sunspot)],
                        'active': True
                    }
                    self.sunspot_tracks.append(track)
        
        # Filter tracks with at least 2 observations
        self.sunspot_tracks = [t for t in self.sunspot_tracks if len(t['observations']) >= 2]
        
        print(f"Created {len(self.sunspot_tracks)} sunspot tracks")
    
    def calculate_rotation_rates(self):
        """
        Calculate rotation rates for tracked sunspots
        
        Returns:
        --------
        rotation_data : list
            List of dictionaries with rotation information
        """
        
        rotation_data = []
        
        for track in self.sunspot_tracks:
            if len(track['observations']) < 2:
                continue
            
            times = []
            longitudes = []
            latitudes = []
            
            for obs_time, sunspot in track['observations']:
                times.append(obs_time.to_value(u.day))
                longitudes.append(float(sunspot['heliographic_lon'].to_value(u.deg)))
                latitudes.append(float(sunspot['heliographic_lat'].to_value(u.deg)))
            
            times = np.array(times)
            longitudes = np.array(longitudes)
            latitudes = np.array(latitudes)
            
            # Handle longitude wrapping
            longitudes = self._unwrap_longitude(longitudes)
            
            # Fit linear trend to longitude vs time
            if len(times) >= 2:
                try:
                    # Linear fit
                    poly_coeffs = np.polyfit(times, longitudes, 1)
                    rotation_rate_deg_day = poly_coeffs[0]
                    
                    # Convert to angular velocity in deg/day
                    mean_latitude = np.mean(latitudes)
                    
                    track_data = {
                        'track_id': track['track_id'],
                        'mean_latitude': mean_latitude,
                        'rotation_rate_deg_day': rotation_rate_deg_day,
                        'rotation_rate_microrad_s': rotation_rate_deg_day * np.pi / 180 / 86400 * 1e6,
                        'observations_count': len(track['observations']),
                        'time_span_days': times[-1] - times[0],
                        'longitude_change': longitudes[-1] - longitudes[0],
                        'times': times,
                        'longitudes': longitudes,
                        'latitudes': latitudes
                    }
                    
                    rotation_data.append(track_data)
                    
                except Exception as e:
                    print(f"Error calculating rotation for track {track['track_id']}: {e}")
        
        return rotation_data
    
    def _unwrap_longitude(self, longitudes):
        """Handle longitude wrapping around 360 degrees"""
        unwrapped = np.unwrap(np.radians(longitudes))
        return np.degrees(unwrapped)
    
    def fit_differential_rotation(self, rotation_data):
        """
        Fit differential rotation law to observed rotation rates
        
        Parameters:
        -----------
        rotation_data : list
            Output from calculate_rotation_rates()
        
        Returns:
        --------
        fit_params : dict
            Fitted differential rotation parameters
        """
        
        if len(rotation_data) < 3:
            print("Need at least 3 tracks for differential rotation fit")
            return None
        
        latitudes = np.array([r['mean_latitude'] for r in rotation_data])
        rotation_rates = np.array([r['rotation_rate_deg_day'] for r in rotation_data])
        
        # Convert latitude to radians
        lat_rad = np.radians(np.abs(latitudes))
        
        # Differential rotation law: ω(θ) = A + B*sin²(θ) + C*sin⁴(θ)
        def diff_rot_model(lat_rad, A, B, C):
            return A + B * np.sin(lat_rad)**2 + C * np.sin(lat_rad)**4
        
        try:
            # Fit the model
            popt, pcov = curve_fit(diff_rot_model, lat_rad, rotation_rates,
                                 p0=[14.7, -2.4, -1.8])  # Initial guess based on known values
            
            A, B, C = popt
            errors = np.sqrt(np.diag(pcov))
            
            # Calculate R-squared
            ss_res = np.sum((rotation_rates - diff_rot_model(lat_rad, *popt)) ** 2)
            ss_tot = np.sum((rotation_rates - np.mean(rotation_rates)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            fit_params = {
                'A': A,
                'B': B, 
                'C': C,
                'A_error': errors[0],
                'B_error': errors[1],
                'C_error': errors[2],
                'r_squared': r_squared,
                'equatorial_rate': A,  # deg/day
                'equatorial_period': 360.0 / A,  # days
            }
            
            print("Differential Rotation Fit Results:")
            print(f"ω(θ) = {A:.3f} + {B:.3f}*sin²(θ) + {C:.3f}*sin⁴(θ) deg/day")
            print(f"Equatorial rotation rate: {A:.3f} ± {errors[0]:.3f} deg/day")
            print(f"Equatorial period: {360.0/A:.2f} days")
            print(f"R-squared: {r_squared:.4f}")
            
            return fit_params
            
        except Exception as e:
            print(f"Error fitting differential rotation: {e}")
            return None
    
    def visualize_tracking_results(self, rotation_data, fit_params=None):
        """
        Visualize sunspot tracking and rotation analysis results
        
        Parameters:
        -----------
        rotation_data : list
            Output from calculate_rotation_rates()
        fit_params : dict, optional
            Output from fit_differential_rotation()
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sunspot tracks in longitude vs time
        ax1 = axes[0, 0]
        for track_data in rotation_data:
            times_rel = track_data['times'] - track_data['times'][0]
            ax1.plot(times_rel, track_data['longitudes'], 'o-', 
                    label=f"Track {track_data['track_id']}")
        
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Longitude (degrees)')
        ax1.set_title('Sunspot Longitude Evolution')
        ax1.grid(True, alpha=0.3)
        if len(rotation_data) <= 10:
            ax1.legend()
        
        # Plot 2: Rotation rate vs latitude
        ax2 = axes[0, 1]
        latitudes = [r['mean_latitude'] for r in rotation_data]
        rotation_rates = [r['rotation_rate_deg_day'] for r in rotation_data]
        
        ax2.scatter(latitudes, rotation_rates, c='blue', s=50, alpha=0.7)
        
        # Plot theoretical curve if fit parameters available
        if fit_params:
            lat_theory = np.linspace(-60, 60, 100)
            lat_rad_theory = np.radians(np.abs(lat_theory))
            rate_theory = (fit_params['A'] + 
                          fit_params['B'] * np.sin(lat_rad_theory)**2 + 
                          fit_params['C'] * np.sin(lat_rad_theory)**4)
            ax2.plot(lat_theory, rate_theory, 'r-', linewidth=2, 
                    label='Fitted curve')
            ax2.legend()
        
        ax2.set_xlabel('Latitude (degrees)')
        ax2.set_ylabel('Rotation Rate (deg/day)')
        ax2.set_title('Solar Differential Rotation')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Individual track quality
        ax3 = axes[1, 0]
        track_ids = [r['track_id'] for r in rotation_data]
        time_spans = [r['time_span_days'] for r in rotation_data]
        obs_counts = [r['observations_count'] for r in rotation_data]
        
        scatter = ax3.scatter(time_spans, obs_counts, 
                            c=track_ids, s=50, cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Time Span (days)')
        ax3.set_ylabel('Number of Observations')
        ax3.set_title('Track Quality Assessment')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Track ID')
        
        # Plot 4: Residuals from fit (if available)
        ax4 = axes[1, 1]
        if fit_params:
            lat_rad = np.radians(np.abs(latitudes))
            predicted_rates = (fit_params['A'] + 
                              fit_params['B'] * np.sin(lat_rad)**2 + 
                              fit_params['C'] * np.sin(lat_rad)**4)
            residuals = np.array(rotation_rates) - predicted_rates
            
            ax4.scatter(latitudes, residuals, c='red', s=50, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Latitude (degrees)')
            ax4.set_ylabel('Residuals (deg/day)')
            ax4.set_title('Fit Residuals')
            ax4.grid(True, alpha=0.3)
            
            # Add RMS error text
            rms_error = np.sqrt(np.mean(residuals**2))
            ax4.text(0.05, 0.95, f'RMS Error: {rms_error:.3f} deg/day', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No fit available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Fit Residuals')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, rotation_data, filename='sunspot_rotation_results.csv'):
        """
        Export rotation analysis results to CSV
        
        Parameters:
        -----------
        rotation_data : list
            Output from calculate_rotation_rates()
        filename : str
            Output filename
        """
        
        df_data = []
        for track_data in rotation_data:
            df_data.append({
                'track_id': track_data['track_id'],
                'mean_latitude': track_data['mean_latitude'],
                'rotation_rate_deg_day': track_data['rotation_rate_deg_day'],
                'rotation_rate_microrad_s': track_data['rotation_rate_microrad_s'],
                'observations_count': track_data['observations_count'],
                'time_span_days': track_data['time_span_days'],
                'longitude_change': track_data['longitude_change']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        
        return df


class DifferentialRotationAnalyzer:
    """Comprehensive analysis class for solar differential rotation"""
    
    def __init__(self):
        self.tracker = SunspotTracker()
        
    def analyze_image_sequence(self, image_data_list, observation_times):
        """
        Analyze a sequence of solar images for differential rotation
        
        Parameters:
        -----------
        image_data_list : list
            List of tuples (image_path, sunspots, disk_info)
        observation_times : list
            List of observation times
        
        Returns:
        --------
        analysis_results : dict
            Complete analysis results
        """
        
        print("Starting differential rotation analysis...")
        
        # Add all observations to tracker
        for i, (image_path, sunspots, disk_info) in enumerate(image_data_list):
            self.tracker.add_observation(sunspots, observation_times[i], disk_info)
        
        # Match sunspots across observations
        print("Matching sunspots across observations...")
        self.tracker.match_sunspots()
        
        # Calculate rotation rates
        print("Calculating rotation rates...")
        rotation_data = self.tracker.calculate_rotation_rates()
        
        if not rotation_data:
            print("No valid rotation data calculated")
            return None
        
        # Fit differential rotation law
        print("Fitting differential rotation law...")
        fit_params = self.tracker.fit_differential_rotation(rotation_data)
        
        # Visualize results
        print("Creating visualizations...")
        self.tracker.visualize_tracking_results(rotation_data, fit_params)
        
        # Export results
        df = self.tracker.export_results(rotation_data)
        
        analysis_results = {
            'rotation_data': rotation_data,
            'fit_parameters': fit_params,
            'tracks': self.tracker.sunspot_tracks,
            'results_dataframe': df
        }
        
        return analysis_results
    
    def compare_with_literature(self, fit_params):
        """
        Compare fitted parameters with literature values
        
        Parameters:
        -----------
        fit_params : dict
            Fitted differential rotation parameters
        """
        
        if not fit_params:
            print("No fit parameters to compare")
            return
        
        # Literature values (Snodgrass & Ulrich 1990)
        literature_A = 14.713  # deg/day
        literature_B = -2.396  # deg/day
        literature_C = -1.787  # deg/day
        
        print("\nComparison with Literature Values:")
        print("=" * 50)
        print(f"Parameter A (equatorial rate):")
        print(f"  Your fit:    {fit_params['A']:.3f} ± {fit_params['A_error']:.3f} deg/day")
        print(f"  Literature:  {literature_A:.3f} deg/day")
        print(f"  Difference:  {fit_params['A'] - literature_A:.3f} deg/day")
        
        print(f"\nParameter B:")
        print(f"  Your fit:    {fit_params['B']:.3f} ± {fit_params['B_error']:.3f} deg/day")
        print(f"  Literature:  {literature_B:.3f} deg/day")
        print(f"  Difference:  {fit_params['B'] - literature_B:.3f} deg/day")
        
        print(f"\nParameter C:")
        print(f"  Your fit:    {fit_params['C']:.3f} ± {fit_params['C_error']:.3f} deg/day")
        print(f"  Literature:  {literature_C:.3f} deg/day")
        print(f"  Difference:  {fit_params['C'] - literature_C:.3f} deg/day")
        
        # Calculate chi-squared
        chi_sq = (((fit_params['A'] - literature_A) / fit_params['A_error'])**2 + 
                  ((fit_params['B'] - literature_B) / fit_params['B_error'])**2 + 
                  ((fit_params['C'] - literature_C) / fit_params['C_error'])**2)
        
        print(f"\nChi-squared (3 DOF): {chi_sq:.2f}")
        
        # Physical interpretation
        equatorial_period = 360.0 / fit_params['A']
        pole_rate = fit_params['A'] + fit_params['B'] + fit_params['C']  # at 90° latitude
        pole_period = 360.0 / pole_rate if pole_rate > 0 else np.inf
        
        print(f"\nPhysical Interpretation:")
        print(f"Equatorial period: {equatorial_period:.2f} days")
        print(f"Polar period: {pole_period:.2f} days" if pole_period != np.inf else "Polar period: undefined")
        print(f"Period difference: {pole_period - equatorial_period:.2f} days" if pole_period != np.inf else "Period difference: large")