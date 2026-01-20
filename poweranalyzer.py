#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:50:58 2025

@author: pi

Power Event Analyzer - Analyzes power events, clusters similar events, and converts labeled clusters to devices.

Naming convention:
- Events: Individual power events (rows in the dataframe with timestamp and power measurements)
- Clusters: Groups of similar events (from K-means clustering)
- Devices: When a cluster is labeled, it becomes a device

after creating and removing devices, suggest you use mqtt-explorer.com to clean up unused topics.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans    #pip install scikit-learn
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import os
import logging
from database import PowerEventDatabase

# Set up logger
logger = logging.getLogger(__name__)

EVENT_SIZE = 20 # and event is 20 samples (20 seconds)
MIN_CLUSTERS = 3 #3
MAX_CLUSTERS = 100 #100
MIN_DISTANCE = 200 

def myscaler(data):
    #columns are the power values
    #smoothing to remove any sub 1 sample offsets
    smoothed = (data[:, :-1] + data[:, 1:]) / 2
    #each_event_max = smoothed.max(axis=1)
    #each_event_max_exp = each_event_max[:,None];
    # each_event_min = smoothed.min(axis=1)
    #each_event_min_exp = each_event_min[:,None]
    #on_or_off = (np.abs(each_event_min) < np.abs(each_event_max))
    #actually only if there is a sign change
    #on_events = smoothed + np.where(each_event_min_exp < 0, np.abs(each_event_min_exp), 0)
    #off_events = np.where(each_event_max_exp > 0, np.abs(each_event_max_exp), 0) - smoothed
    #scaled = np.where(on_or_off[:, None], on_events, off_events)
    #scaled = np.log1p(scaled) / np.log1p(maxpower)
    #scaled = smoothed/maxpower
    #scaled = np.where(on_or_off[:, None], scaled, -scaled)
    return smoothed

class PowerEventAnalyzer:
    def __init__(self):
        """
        Initialize the Power Event Analyzer

        parameters:
        timestamps: numpy array of timestamps
        events: numpy array of events
        labels: numpy array of labels
        profiles: numpy array of profiles
        distances: numpy array of distances
        off_delays: numpy array of off_delays
        
        Naming convention:
        - Events: Individual power events 
        - Clusters: Groups of similar events (from K-means clustering)
        - Devices: When a cluster is labeled, it becomes a device
        """
        # storage for events
        self.timestamps = np.array([], dtype=object)
        self.events = np.zeros((0,EVENT_SIZE), dtype=float)
        self.scaled_events = myscaler(self.events)
        self.extreme_powers = self.events.max(axis=1)
        self.n_events = 0
        self.first_event = None
        self.last_event = None
        self.clusters = -1*np.ones(len(self.timestamps),dtype=int) # map of the cluster assigned to each event
        self.devices = -1*np.ones(len(self.timestamps),dtype=int) # map of the device assigned to each event
        # firs step of analysis is to create clusters of events
        self.n_clusters = None
        self.cluster_medians = np.zeros((0, EVENT_SIZE)) # the profile (median pattern) of each cluster
        self.max_cluster_distances = np.full(0,np.inf) # the maximum distance within each cluster
        # second step is to map the clusters to devices
        self.n_devices = 0
        self.date_of_analysis_in_memory = None
        self.device_keys = np.array([], dtype=int) # keys for each device (from database)
        self.device_labels = np.array([], dtype=str) # names for each device (labeled clusters become devices)
        self.device_profiles = np.zeros((0, EVENT_SIZE)) # profile for each device (from cluster medians when labeled)
        self.scaled_device_profiles = myscaler(self.device_profiles)
        self.max_device_distances = np.full(0,np.inf) # the maximum distance within each device
        self.off_delays = -1*np.ones(0,dtype=int)        # Ensure all devices have an off_delay (default to -1 if missing)
        
        # Window position storage
        self.window_pos_timeline = None
        self.window_pos_power = None
        self.window_pos_histogram = None
        
        # Figure references for plot functions to manage their own lifecycle
        self.fig_timeline = None
        self.fig_power = None
        self.fig_histogram = None
        # database connection
        self.db = None
        self._db_init()


    def _db_init(self):
        """Initialize the database"""
        self.db = PowerEventDatabase()

    def close(self):
        """Close the database"""
        self.db.close()
        self.db = None
        

    def get_status(self):
        """Get the status of the database"""
        status = self.db.get_status()
        self.n_events = status['number_of_events']
        self.date_of_analysis_in_memory = status['latest_analysis']
        self.first_event = status['first_event']
        self.last_event = status['last_event']
        self.n_devices = status['n_devices']
        return status

    def load_events(self):
        """Load events from the database"""
        self.timestamps, self.events = self.db.load_data(existing_timestamps=self.timestamps, existing_events=self.events)
        if len(self.events) > 0:
            self.scaled_events = myscaler(self.events)
            self.extreme_powers = self.events.max(axis=1)
            min_powers = self.events.min(axis=1)
            self.extreme_powers = np.where(np.abs(self.extreme_powers) >= np.abs(min_powers), self.extreme_powers, min_powers)
            self.clusters = -1*np.ones(len(self.timestamps),dtype=int) # map of the cluster assigned to each event
            self.devices = -1*np.ones(len(self.timestamps),dtype=int) # map of the device assigned to each event
            self.n_events = len(self.timestamps)
            self.first_event = self.timestamps[0]
            self.last_event = self.timestamps[-1]


    def _determine_optimal_clusters(self, max_clusters=MAX_CLUSTERS, unlabeled=False):
        """Determine optimal number of clusters using MIN_DISTANCE with binary search"""
        if unlabeled:
            events = self.events
            scaled_events = myscaler(events)
        else:
            mask = self.devices == -1
            events = self.events[mask]
            scaled_events = myscaler(events)
        if max_clusters == MIN_CLUSTERS:
            return MAX_CLUSTERS    
        
        # Binary search bounds
        low = MIN_CLUSTERS
        high = min(max_clusters, len(events))
        best_k = len(events)//100
        
        print('Finding optimal number of clusters')
        print('Minimum distance between clusters: ', MIN_DISTANCE)
        print('Finding optimal number of clusters between ', MIN_CLUSTERS, ' and ', max_clusters)
        # Start with average of MIN_CLUSTERS and MAX_CLUSTERS
        k = (low + high)//2
        visited_k = set()  # Track visited k values to avoid infinite loops
        
        while low <= high and k not in visited_k:
            visited_k.add(k)
            
            # Ensure k is within valid range
            k = max(MIN_CLUSTERS, min(k, high))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_events)
            
            # Calculate cluster medians
            cluster_medians = np.zeros((k, EVENT_SIZE))
            for cluster_id in range(k):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = events[cluster_mask]
                cluster_medians[cluster_id] = np.median(cluster_data, axis=0)
            # Scale cluster medians before calculating distances
            scaled_cluster_medians = myscaler(cluster_medians)
            distances = cdist(scaled_cluster_medians, scaled_cluster_medians, metric='euclidean')
            lower_triangle = np.tril(np.ones_like(distances),k=-1)
            minimum_distance = np.min(np.where(lower_triangle, distances, np.inf))
            print(f'Minimum distance between {k} clusters: {minimum_distance:.0f}')    
            
            # Binary search logic
            if minimum_distance < MIN_DISTANCE:
                # Clusters are too close, need fewer clusters
                # Go half as high again (halfway down from current k towards low)
                best_k = k
                # Calculate halfway point between current k and low
                next_k = (k + low)//2
                if next_k < k:
                    high = k - 1
                    k = next_k
                else:
                    break
            else:
                # Clusters are far enough apart, can try more clusters
                best_k = k
                next_k = (k + high)//2
                next_k = min(next_k, high)  # Cap at high
                if next_k > k:
                    low = k + 1
                    k = next_k
                else:
                    break
        return best_k

    def perform_clustering(self, unlabeled=False):
        if unlabeled:
            print(f'Using {len(self.events[self.devices == -1])} unlabeled events for clustering')
            scaled_events = self.scaled_events[self.devices == -1]
            events = self.events[self.devices == -1]
        else:
            scaled_events = self.scaled_events
            events = self.events
            print(f'Using all {len(self.events)} events for clustering')
        if len(scaled_events) < 10:
            logger.error("Not enough events for clustering")
            return None
        self.n_clusters = self._determine_optimal_clusters(unlabeled=unlabeled)
        print('Performing clustering with ', self.n_clusters, ' clusters')
        """Perform K-means clustering on power data"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_events).astype(int)
        
        # Calculate cluster medians
        self.cluster_medians = np.zeros((self.n_clusters, EVENT_SIZE))
        self.max_cluster_distances = np.full(self.n_clusters, np.inf)
        for cluster_id in range(self.n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = events[cluster_mask]
            cluster_median = np.median(cluster_data, axis=0)
            self.cluster_medians[cluster_id] = cluster_median   
            # Scale cluster data and median before calculating distances
            scaled_cluster_data = myscaler(cluster_data)
            cluster_median_scaled = myscaler(cluster_median.reshape(1, -1))
            distances = cdist(scaled_cluster_data, cluster_median_scaled, metric='euclidean')
            self.max_cluster_distances[cluster_id] = np.max(distances)
        if unlabeled:
            logger.info("need to pulll in rest of the events")
            self._reassign_devices_from_medians()
        else:
            self.clusters = clusters


    def plot_day_of_week_histogram(self, mask, title_label=None):
        """
        Plot histogram showing number of occurrences for each day of the week
        for events matching the provided mask.
        
        Args:
            mask: Boolean array indicating which events to include
            title_label: Optional label to include in the title
            
        Returns:
            fig, ax: matplotlib figure and axis objects
        """
        # Close previous histogram figure if it exists
        if self.fig_histogram is not None:
            self.save_window_position(self.fig_histogram, 'histogram')
            plt.close(self.fig_histogram)
        
        
        # Get day of week for filtered events (0=Monday, 6=Sunday)
        days_of_week = np.array([ts.weekday() for ts in self.timestamps[mask]], dtype=int)
        
        # Count occurrences for each day
        day_counts = np.bincount(days_of_week, minlength=7)
        
        # Count number of unique dates for each day of the week
        # Convert timestamps to date (year-month-day) to get unique dates
        filtered_dates = np.array([ts.date() for ts in self.timestamps[mask]], dtype=object)
        unique_dates_per_day = np.zeros(7, dtype=int)
        
        for day_idx in range(7):
            # Get all events for this day of week
            day_mask = days_of_week == day_idx
            if day_mask.sum() > 0:
                # Get unique dates for this day of week
                dates_for_day = filtered_dates[day_mask]
                unique_dates_per_day[day_idx] = len(np.unique(dates_for_day))
        
        # Calculate occurrences per day (divide by number of unique dates for that day)
        # Avoid division by zero
        occurrences_per_day = np.zeros(7, dtype=float)
        for day_idx in range(7):
            if unique_dates_per_day[day_idx] > 0:
                occurrences_per_day[day_idx] = day_counts[day_idx] / unique_dates_per_day[day_idx]
            else:
                occurrences_per_day[day_idx] = 0.0
        
        # Day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create figure and axis
        fig, ax = plt.subplots()
        self.fig_histogram = fig
        
        # Create bar plot using numeric positions to avoid categorical axis issues
        x_pos = np.arange(len(day_names))
        bars = ax.bar(x_pos, occurrences_per_day, color='steelblue', alpha=0.7)
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(day_names)
        
        # Add value labels on top of bars
        for bar, count in zip(bars, occurrences_per_day):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(occurrences_per_day)*0.01,
                       f'{count:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Occurrences per Day')
        
        # Set title
        if title_label:
            ax.set_title(f'Day of Week Distribution {title_label}')
        else:
            ax.set_title('Day of Week Distribution')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self.restore_window_position(fig, 'histogram') 
        if plt.isinteractive():
            plt.show(block=False)
        return fig, ax
    

    def plot_timeline(self, mask=None, title_label=None):
        """
        Plot timeline of power events
        
        Args:
            mask: Boolean array indicating which events to include. If None, creates default mask (all events or random 1000 if > 1000)
            title_label: Optional label to include in the title
        """
        # Close previous timeline figure if it exists
        if self.fig_timeline is not None:
            self.save_window_position(self.fig_timeline, 'timeline')
            plt.close(self.fig_timeline)
        
        fig, ax = plt.subplots()
        self.fig_timeline = fig

        # Create default mask if not provided
        if mask is None:
            n_points = len(self.extreme_powers)
            if n_points > 1000:
                # Randomly choose 1000 indices for timeline
                selected = np.random.choice(n_points, size=1000, replace=False)
                mask = np.zeros(n_points, dtype=bool)
                mask[selected] = 1
            else:
                mask = np.ones(n_points, dtype=bool)
        # Convert timestamps to time of day
        times_of_day = np.array([ts.time() for ts in self.timestamps], dtype=object)


        # Convert time objects to matplotlib float representation (hours since midnight)
        times_float = np.array([t.hour + t.minute/60 + t.second/3600 for t in times_of_day])

        # Create scatter plot of extreme power values        
        scatter = ax.scatter(
            times_float[mask], 
            self.extreme_powers[mask],
            s=50,
            alpha=0.7
        )

        # Draw vertical lines from zero to each point and annotate with label
        # Label events if they are matched to a device
        for t_float, power, cluster_id, device_idx in zip(times_float[mask], 
                                                        self.extreme_powers[mask], 
                                                        self.clusters[mask], 
                                                        self.devices[mask]):
            # Prioritize device labels over cluster labels
            if device_idx != -1:
                label = self.device_labels[device_idx]
            elif cluster_id != -1:
                label = self._get_device_label_for_cluster(cluster_id)
            else:
                label = None
            
            if label is not None:
                ax.plot([t_float, t_float], [0, power],  
                         alpha=0.5, linewidth=2, label=label)
                ax.annotate(
                    label,
                    (t_float, power),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='left',
                    fontsize=10,
                    color='black',
                    weight='bold'
                )

        # Formatting for time-of-day axis
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3) 
        ax.set_xlabel('Time of Day') 
        ax.set_ylabel('Extreme Power Value') 
        if title_label:
            ax.set_title(f'Power of {title_label}')
        else:
            ax.set_title('Power of Events') 
        ax.grid(True, alpha=0.3) 

        # Set x-axis from midnight to midnight
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])       

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        # Restore or set window position
        self.restore_window_position(fig, 'timeline')
        if plt.isinteractive():
            plt.show(block=False)
        return fig, ax
    
    
    def save_results_to_database(self):
        """Save analysis results (devices) to the database.
        
        Args:
            db: PowerEventDatabase instance
        
        Returns:
            bool: True if successful, False otherwise
        """
        n_devices = len(self.device_labels)
        if n_devices == 0:
            print("❌ No devices to save.")
            return False
        
        analysis_data = {
            'device_key': np.arange(n_devices, dtype=int),
            'device_label': self.device_labels,
            'max_device_distance': self.max_device_distances,
            'off_delay': self.off_delays,
            'profile': self.device_profiles
        }
        
        success = self.db.save_analysis(analysis_data)
        if success:
            logger.info(f"✅ Saved {n_devices} devices to database")
        else:
            logger.error(f"❌ Failed to save devices to database.")
            return False
            
    def load_results_from_database(self):
        """Load analysis results (devices) from the database.
        
        Args:
            db: PowerEventDatabase instance
        Returns:
            bool: True if successful, False otherwise
        """
        
        analysis_data = self.db.load_analysis()
        if analysis_data is None:
            logger.error("❌ No devices found in database.")
            return None
        
        # Extract data
        self.device_keys = np.array(analysis_data['device_key'], dtype=int)
        self.device_labels = np.array(analysis_data['device_label'])
        self.max_device_distances = np.array(analysis_data['max_device_distance'], dtype=float)
        self.off_delays = np.array(analysis_data['off_delay'], dtype=int)
        self.device_profiles = np.array(analysis_data['profile'], dtype=float)
        self.scaled_device_profiles = myscaler(self.device_profiles)
        status = self.db.get_status()
        self.date_of_analysis_in_memory = status['latest_analysis']
        
        logger.info(f"✅ Loaded {len(self.device_labels)} devices from database")
        
        # Reassign clusters to all events based on similarity to cluster medians
        if self.events is not None:
           logger.info("🔄 Reassigning devices to all events based on loaded patterns...")
           self._reassign_devices_from_medians()

        self.display_devices_report()

        return self.update_mqtt_topics()

    def update_mqtt_topics(self):
        """Update MQTT topics for all devices."""
        binary_topics = {}
        value_topics = {}
        mqtt_topic_prefix = "powermonitor"
        binary_postfix = "_power_status"
        value_postfix = "_power"
        for device in np.unique(self.device_labels):
            new_device = device.replace(" ", "_").lower()
            binary_topics[new_device] = {
                    "config_topic": f"homeassistant/binary_sensor/{mqtt_topic_prefix}/{new_device}{binary_postfix}/config",
                    "state_topic": f"homeassistant/binary_sensor/{mqtt_topic_prefix}/{new_device}{binary_postfix}/state",
                    "icon": "mdi:flash",
                    "name": f'{new_device}{binary_postfix}',
                    "unique_id": f'{new_device}{binary_postfix}',
                    "device_class": "power",
                    "device": {
                        "name": "Power Monitor",
                        "model": "Power Monitor 1.0",
                        "manufacturer": "nils154",
                        "identifiers": 'Power Monitor',
                    }
                }   
            value_topics[f'{device}_power'] = {
                    "config_topic": f"homeassistant/sensor/{mqtt_topic_prefix}/{new_device}{value_postfix}/config",
                    "state_topic": f"homeassistant/sensor/{mqtt_topic_prefix}/{new_device}{value_postfix}/state",
                    "icon": "mdi:flash",
                    "name": f'{new_device}{value_postfix}',
                    "unique_id": f'{new_device}{value_postfix}',
                    "device_class": "power",
                    "unit_of_measurement": "W",
                    "device": {
                        "name": "Power Monitor",
                        "model": "Power Monitor 1.0",
                        "manufacturer": "nils154",
                        "identifiers": 'Power Monitor',
                    }
                }     
        return binary_topics|value_topics

    def match(self, timeStamp, this_event_power_data, avg_power, hassmqtt):
        """
        Match event to a device and handle MQTT notifications.
        
        Args:
            timeStamp: Unix timestamp for the event 
            this_event_power_data: numpy array of power values (EVENT_SIZE length)
            avg_power: Average power value 
            hassmqtt: HassMQTT instance
        Returns:
            tuple: (device_label, scheduled_time)
                   or (None, None) if no match
        """
        # Check for new analysis and reload if needed
        status = self.db.get_status()
        analysis_date = status['latest_analysis']
        if analysis_date is not None and self.date_of_analysis_in_memory < analysis_date:
            logger.info('new analysis found')
            updated_topics = self.load_results_from_database()
            hassmqtt.reconfig(updated_topics)
        if self.n_devices == 0:
            logger.error("No devices found, cannot match event")
            return None, None
        scaled_power_data = myscaler(this_event_power_data.reshape(1, -1))  # make it 2D, 1 row, infer column
        distance = cdist(scaled_power_data, self.scaled_device_profiles, metric='euclidean')
        min_distance = np.min(distance)   
        best_device = np.argmin(distance)
        message = "ON" if avg_power > 0 else "OFF"
        
        if min_distance < self.max_device_distances[best_device]:
            device_label = self.device_labels[best_device]
            off_delay = self.off_delays[best_device]
            logger.info(f'Matched to device #{best_device}:{device_label} turning {message} with distance:{min_distance:.3f}')
            
            # Handle MQTT operations if dependencies provided
            hassmqtt.update(device_label, message, False)
            power_value = avg_power if avg_power > 0 else 0
            hassmqtt.update(f'{device_label}_power', str(power_value), False)
            # Queue off action if message is ON and avg_power > 0 and off_delay > 0
            if message == "ON" and off_delay >= 0:
                    scheduled_time = timeStamp + timedelta(minutes=int(off_delay))  
                    logger.info(f"Queued OFF action for {device_label} in {off_delay} minutes, an {'int' if isinstance(off_delay, int) else 'float'}.")
            else:   
                scheduled_time = None
            return device_label, scheduled_time
        else:
            return None, None

    def _get_device_id_for_cluster(self, cluster_id):
        """
        Get the device index for a cluster if it exists as a device.
        Returns device index if found, None otherwise.
        """
        cluster_median = self.cluster_medians[cluster_id]
        # Iterate through device profiles array
        for dev_idx in range(len(self.device_profiles)):
            profile = self.device_profiles[dev_idx]
            if np.array_equal(profile, cluster_median):
                return dev_idx  # Return device index
        return None
    
    def _get_device_label_for_cluster(self, cluster_id):
        """
        Get the device label for a cluster if it exists as a device.
        Returns label if found, None otherwise.
        """
        device_idx = self._get_device_id_for_cluster(cluster_id)
        if device_idx is not None and device_idx < len(self.device_labels):
            return self.device_labels[device_idx]
        return None
    
    def _is_cluster_a_device(self, cluster_id):
        """
        Check if a cluster is already a device.
        Returns True if the cluster median exists in device_profiles.
        """
        return self._get_device_id_for_cluster(cluster_id) is not None
    
    def modify_device(self, device_idx, device_label=None, max_device_distance=None, off_delay=None):
        """
        Modify an existing device in both memory and database.
        
        This method wraps db.modify_device and also updates in-memory arrays.
        
        Args:
            device_idx: Integer device index (array index, not database key)
            device_label: Optional string device label to update
            max_device_distance: Optional float max device distance to update
            off_delay: Optional integer off delay to update
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            IndexError: If device_idx is out of range

        """
        if device_idx < 0 or device_idx >= len(self.device_keys):
            raise IndexError(f"device_idx {device_idx} is out of range (0-{len(self.device_keys)-1})")
        
        # Get device_key from device_idx
        device_key = self.device_keys[device_idx]
        
        # Call database method
        success = self.db.modify_device(device_key, device_label=device_label, 
                                       max_device_distance=max_device_distance, 
                                       off_delay=off_delay)
        
        if success:
            # Update in-memory arrays
            if device_label is not None:
                self.device_labels[device_idx] = device_label
            
            if max_device_distance is not None:
                self.max_device_distances[device_idx] = float(max_device_distance)
                # Recalculate scaled profiles if needed
                if len(self.device_profiles) > 0:
                    self.scaled_device_profiles = myscaler(self.device_profiles)
                # Reassign devices if events are loaded
                if self.events is not None and len(self.events) > 0:
                    self._reassign_devices_from_medians()
            
            if off_delay is not None:
                self.off_delays[device_idx] = int(off_delay)
        return success
    
    def delete_device(self, device_idx):
        """
        Delete a device from both memory and database.
        
        This method wraps db.delete_device and also updates in-memory arrays.
        
        Args:
            device_idx: Integer device index (array index, not database key)
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            IndexError: If device_idx is out of range
        """
        if device_idx < 0 or device_idx >= len(self.device_keys):
            raise IndexError(f"device_idx {device_idx} is out of range (0-{len(self.device_keys)-1})")
        
        # Get device_key from device_idx
        device_key = self.device_keys[device_idx]
        
        # Call database method
        success = self.db.delete_device(device_key)
        
        if success:
            # Remove device from all in-memory arrays
            # Use numpy delete to remove the element at device_idx
            self.device_keys = np.delete(self.device_keys, device_idx)
            self.device_labels = np.delete(self.device_labels, device_idx)
            self.device_profiles = np.delete(self.device_profiles, device_idx, axis=0)
            self.max_device_distances = np.delete(self.max_device_distances, device_idx)
            self.off_delays = np.delete(self.off_delays, device_idx)
            self.n_devices -= 1
            # Recalculate scaled profiles
            if len(self.device_profiles) > 0:
                self.scaled_device_profiles = myscaler(self.device_profiles)
            else:
                self.scaled_device_profiles = np.zeros((0, EVENT_SIZE))
            
            # Update devices array: set all references to this device_idx to -1
            # Also shift down indices for devices after the deleted one
            #(instead of self._reassign_devices_from_medians())
            if self.devices is not None and len(self.devices) > 0:
                # Set references to deleted device to -1
                self.devices[self.devices == device_idx] = -1
                # Shift down indices for devices after the deleted one
                mask = self.devices > device_idx
                self.devices[mask] = self.devices[mask] - 1
            
        return success
    
    def _convert_cluster_to_device(self, cluster_id, label):
        """
        Convert a labeled cluster to a device by adding it to device arrays.
        If the cluster is already a device (same profile exists), update the existing device.
        Returns the device_idx (existing or new).
        
        Args:
            cluster_id: The cluster ID to convert
            label: The label for the device
            db: Optional PowerEventDatabase instance to add device to database
        """
        # Check if this cluster is already a device (was labeled before)
        existing_device_idx = self._get_device_id_for_cluster(cluster_id)
        
        if existing_device_idx is not None:
            print(f"Cluster {cluster_id} is already a device: {existing_device_idx}")
            print('edit the device instead')
            return existing_device_idx
        else:
            # Create new device - append to device arrays
            next_device_idx = len(self.device_labels)
            
            # Determine device_key
            device_key = max(self.device_keys) + 1 if len(self.device_keys) > 0 else 0
            profile = self.cluster_medians[cluster_id]
            # Append to device arrays
            self.device_keys = np.append(self.device_keys, device_key)
            self.device_labels = np.append(self.device_labels, label)
            self.device_profiles = np.vstack([self.device_profiles, profile.reshape(1, -1)])
            self.max_device_distances = np.append(self.max_device_distances, self.max_cluster_distances[cluster_id])
            self.off_delays = np.append(self.off_delays, np.array([-1], dtype=int))  # Default off_delay
            self.scaled_device_profiles = myscaler(self.device_profiles)
            self.n_devices += 1
            self.db.add_device(device_key, label, self.max_cluster_distances[cluster_id], -1, profile)
            
            return next_device_idx
    
    def add_device(self, event_index, match_distance, label):
        """
        Add a new device based on matching events at a specific event index.
        Finds all events matching the event at event_index using match_distance,
        calculates the median pattern, and creates a new device.
        
        Args:
            event_index: Index of the event to use as the reference
            match_distance: Distance threshold for matching events
            label: Label/name for the new device
            
        Returns:
            int: The device index of the newly created device
        """
        if event_index < 0 or event_index >= len(self.events):
            raise ValueError(f"Invalid event_index: {event_index}")
        
        if match_distance <= 0:
            raise ValueError(f"match_distance must be greater than 0, got {match_distance}")
        
        if not label or not label.strip():
            raise ValueError("Device label cannot be empty")
        
        # Get the reference event
        reference_event = self.events[event_index]
        
        # Find all events matching this event using the match_distance
        reference_scaled = myscaler(reference_event.reshape(1, -1))
        distances = cdist(reference_scaled, self.scaled_events, metric='euclidean')[0]
        matching_mask = distances < match_distance
        matching_count = np.sum(matching_mask)
        
        if matching_count == 0:
            # If no matches, just use the reference event itself
            matching_events = reference_event.reshape(1, -1)
        else:
            # Calculate median pattern from matching events
            matching_events = self.events[matching_mask]
        
        median_pattern = np.median(matching_events, axis=0)
        
        # Create new device - append to device arrays
        next_device_idx = len(self.device_labels)
        
        # Determine device_key
        device_key = max(self.device_keys) + 1 if len(self.device_keys) > 0 else 0
        
        # Append to device arrays
        self.device_keys = np.append(self.device_keys, device_key)
        self.device_labels = np.append(self.device_labels, label.strip())
        self.device_profiles = np.vstack([self.device_profiles, median_pattern.reshape(1, -1)])
        self.max_device_distances = np.append(self.max_device_distances, match_distance)
        self.off_delays = np.append(self.off_delays, np.array([-1], dtype=int))  # Default off_delay
        self.scaled_device_profiles = myscaler(self.device_profiles)
        self.n_devices += 1
        
        # Add to database
        self.db.add_device(device_key, label.strip(), match_distance, -1, median_pattern)
        
        # Reassign devices to update assignments
        self._reassign_devices_from_medians()
        
        logger.info(f"✅ Added device '{label}' (device_idx: {next_device_idx}) from {matching_count} matching events at event_index {event_index}")
        
        return next_device_idx
    
    def display_devices_report(self):
        """
        Display a uniform report of all devices with their IDs, labels, and properties.
        """
        if len(self.device_labels) == 0:
            print("No devices found.")
            return
        
        print(f"\nFound {len(self.device_labels)} devices:")
        print("=" * 80)
        
        # Prepare data for table
        rows = []
        has_off_delay = False  # Track if any device has off_delay != -1
        
        for device_idx, label in enumerate(self.device_labels):
            off_delay = self.off_delays[device_idx]
            max_distance = self.max_device_distances[device_idx]
            max_dist_str = f"{max_distance:.0f}" 
            on_off = "ON" if self.device_profiles[device_idx][0] > 0 else "OFF" 
            avg_power = np.mean(self.device_profiles[device_idx])
            
            # Count how many times this device appears in the event data
            count = np.sum(self.devices == device_idx) if hasattr(self, 'devices') and self.devices is not None else 0
            
            row_data = {
                'ID': device_idx,
                'Label': label,
                'ON/OFF': on_off,
                'Avg Power': f"{avg_power:.2f}", 
                'Count': count,
                'Max Distance': max_dist_str
            }
            
            # Only include off_delay if it's not -1
            if off_delay != -1:
                row_data['Off Delay'] = f"{off_delay} min"
                has_off_delay = True
            else:
                row_data['Off Delay'] = None
            
            rows.append(row_data)
        
        # Build column list dynamically
        columns = ['ID', 'Label', 'ON/OFF', 'Avg Power', 'Count']
        if has_off_delay:
            columns.append('Off Delay')
        columns.append('Max Distance')
        
        # Determine column widths
        col_widths = {}
        for col in columns:
            if col == 'Off Delay':
                # Only calculate width for rows that have off_delay
                values = [str(r[col]) for r in rows if r[col] is not None]
            else:
                values = [str(r[col]) for r in rows]
            col_widths[col] = max(len(col), max([len(v) for v in values]) if values else len(col))
        
        # Print header
        header_parts = []
        for col in columns:
            header_parts.append(f"{col:<{col_widths[col]}}")
        header = "  ".join(header_parts)
        
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in rows:
            row_parts = []
            for col in columns:
                if col == 'Off Delay' and row[col] is None:
                    # Skip off_delay column for this row if it's None
                    row_parts.append(" " * col_widths[col])
                else:
                    row_parts.append(f"{row[col]:<{col_widths[col]}}")
            print("  ".join(row_parts))
        
        print("=" * 80)
    
    def _reassign_devices_from_medians(self):
        """
        Reassign all events to devices based on similarity to loaded device profiles
        """
        if len(self.scaled_device_profiles) > 0:
            # Convert cluster_medians dictionary to numpy array
            # cluster_medians is a dict like {0: array, 1: array, 2: array}
            distances = cdist(self.scaled_events, self.scaled_device_profiles, metric = 'euclidean')
            closest_device_distances = np.min(distances, axis=1)
            best_device_indices = np.argmin(distances, axis=1).astype(int)
            max_device_dist_for_best = np.array([self.max_device_distances[idx] for idx in best_device_indices])
            mask = closest_device_distances < max_device_dist_for_best
            self.devices = np.where(mask, best_device_indices, -1).astype(int)
        if len(self.cluster_medians) > 0:
            scaled_cluster_medians = myscaler(self.cluster_medians)
            distances = cdist(self.scaled_events, scaled_cluster_medians, metric='euclidean') #2D array events x clusters
            closest_cluster_distances = np.min(distances, axis=1) #1D array events distances to closest cluster
            best_cluster_indices = np.argmin(distances, axis=1).astype(int) #1D array events indices of closest cluster
            max_cluster_dist_for_best = np.array([self.max_cluster_distances[idx] for idx in best_cluster_indices]) #1D array events max distances to closest cluster
            self.clusters = np.where(closest_cluster_distances < max_cluster_dist_for_best, best_cluster_indices, -1).astype(int) #1D array events clusters indices

            # Print a summary report for each cluster: its device label and how often it was found
            print(f"There are {len(self.cluster_medians)} clusters from the last analysis")
      

    def remove_duplicate_devices(self):
        """
        Compare Devices and remove duplicates
        """
        # Convert device_profiles dictionary to numpy array
        # device_profiles is a dict like {0: array, 1: array, 2: array}
        old_n_devices = len(self.device_labels)
        if old_n_devices > 1:
            # Scale device profiles before calculating distances
            distances = cdist(self.scaled_device_profiles, self.scaled_device_profiles, metric='euclidean')
            lower_triangle = np.tril(np.ones_like(distances),k=-1)
            minimum_distances = np.where(lower_triangle, distances, np.inf)
            mask = np.min(minimum_distances, axis=1) > self.max_device_distances
            self.device_labels = self.device_labels[mask]
            self.device_profiles = self.device_profiles[mask] 
            logger.info(f'Reduced from {old_n_devices} devices to {len(self.device_labels)}')        
           

    def get_cluster_summary(self):
        """Get summary of clusters"""
        summary = []
        for cluster_id in range(self.n_clusters): # type: ignore
            cluster_mask = self.clusters == cluster_id
            count = np.sum(cluster_mask)
            max_distance = self.max_cluster_distances[cluster_id]
            summary.append({
                'cluster_id': cluster_id,
                'count': count,
                'max_distance': max_distance,
            })
        
        # Format as a table string instead of DataFrame
        if not summary:
            return "No clusters found."
        
        # Calculate column widths
        col_widths = {
            'cluster_id': max(len('cluster_id'), max(len(str(s['cluster_id'])) for s in summary)),
            'count': max(len('count'), max(len(str(s['count'])) for s in summary)),
            'max_distance': max(len('max_distance'), max(len(f"{s['max_distance']:.2f}") for s in summary))
        }
        
        # Build header
        header = f"{'cluster_id':<{col_widths['cluster_id']}}  {'count':<{col_widths['count']}}  {'max_distance':<{col_widths['max_distance']}}"
        lines = [header, "-" * len(header)]
        
        # Build rows
        for s in summary:
            max_dist_formatted = f"{s['max_distance']:.2f}"
            row = f"{s['cluster_id']:<{col_widths['cluster_id']}}  {s['count']:<{col_widths['count']}}  {max_dist_formatted:<{col_widths['max_distance']}}"
            lines.append(row)
        
        return "\n".join(lines)
    
    def save_window_position(self, fig, window_name):
        """Save the current window position for a figure"""
        try:
            manager = fig.canvas.manager
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'winfo_geometry'):
                    # Tkinter backend
                    geometry = manager.window.winfo_geometry()
                    # Parse geometry string like "800x600+100+200"
                    if '+' in geometry:
                        size, pos = geometry.split('+')
                        x, y = map(int, pos.split('+'))
                        if window_name == 'timeline':
                            self.window_pos_timeline = (x, y)
                        elif window_name == 'power':
                            self.window_pos_power = (x, y)
                        elif window_name == 'histogram':
                            self.window_pos_histogram = (x, y)
                elif hasattr(manager.window, 'geometry'):
                    # Qt backend
                    geom = manager.window.geometry()
                    x, y, width, height = geom.x(), geom.y(), geom.width(), geom.height()
                    if window_name == 'timeline':
                        self.window_pos_timeline = (x, y, width, height)
                    elif window_name == 'power':
                        self.window_pos_power = (x, y, width, height)
                    elif window_name == 'histogram':
                        self.window_pos_histogram = (x, y, width, height)
        except:
            pass
    
    def restore_window_position(self, fig, window_name):
        """Restore the saved window position for a figure"""
        try:
            manager = fig.canvas.manager
            if hasattr(manager, 'window'):
                if window_name == 'timeline' and self.window_pos_timeline is not None:
                    x, y, width, height = self.window_pos_timeline
                    if hasattr(manager.window, 'wm_geometry'):
                        # Tkinter backend
                        manager.window.wm_geometry(f"+{x}+{y}")
                    elif hasattr(manager.window, 'setGeometry'):
                        # Qt backend
                        geom = manager.window.geometry()
                        manager.window.setGeometry(x, y, geom.width(), geom.height())
                elif window_name == 'power' and self.window_pos_power is not None:
                    x, y, width, height = self.window_pos_power
                    if hasattr(manager.window, 'wm_geometry'):
                        # Tkinter backend
                        manager.window.wm_geometry(f"+{x}+{y}")
                    elif hasattr(manager.window, 'setGeometry'):
                        # Qt backend
                        geom = manager.window.geometry()
                        manager.window.setGeometry(x, y, geom.width(), geom.height())
                elif window_name == 'histogram' and self.window_pos_histogram is not None:
                    x, y, width, height = self.window_pos_histogram
                    if hasattr(manager.window, 'wm_geometry'):
                        # Tkinter backend
                        manager.window.wm_geometry(f"+{x}+{y}")
                    elif hasattr(manager.window, 'setGeometry'):
                        # Qt backend
                        geom = manager.window.geometry()
                        manager.window.setGeometry(x, y, geom.width(), geom.height())
                else:
                    # Default position on left screen if no saved position
                    if window_name == 'timeline':
                        if hasattr(manager.window, 'wm_geometry'):
                            manager.window.wm_geometry("+100+100")
                        elif hasattr(manager.window, 'setGeometry'):
                            manager.window.setGeometry(100, 100, 750, 460)
                    elif window_name == 'power':
                        if hasattr(manager.window, 'wm_geometry'):
                            manager.window.wm_geometry("+100+400")
                        elif hasattr(manager.window, 'setGeometry'):
                            manager.window.setGeometry(900, 100, 800, 560)
                    elif window_name == 'histogram':
                        if hasattr(manager.window, 'wm_geometry'):
                            manager.window.wm_geometry("+100+700")
                        elif hasattr(manager.window, 'setGeometry'):
                            manager.window.setGeometry(100, 700, 600, 460)
        except:
            pass
    
    def plot_power(self, mask, title_label=None, device_pattern=None, highlight_event_index=None, highlight_event_data=None):
        """
        Plot power patterns for events matching the mask, with device labels if available.
        
        Args:
            mask: Boolean array indicating which events to include
            title_label: Optional label to include in the title
            device_pattern: Optional median pattern to plot (if None, calculates from masked events)
            highlight_event_index: Optional index of event to highlight (must be in mask)
            highlight_event_data: Optional event data array to highlight (if provided, used instead of highlight_event_index)
        """
        # Close previous power figure if it exists
        if self.fig_power is not None:
            self.save_window_position(self.fig_power, 'power')
            plt.close(self.fig_power)
        
        fig, ax = plt.subplots()
        self.fig_power = fig
        
        # Reduce to 1000 events if there are more
        masked_indices = np.where(mask)[0]
        if len(masked_indices) > 1000:
            selected_indices = np.random.choice(len(masked_indices), size=1000, replace=False)
            reduced_mask = np.zeros(len(self.events), dtype=bool)
            reduced_mask[masked_indices[selected_indices]] = True
            mask = reduced_mask
            masked_indices = np.where(mask)[0]
        
        power_indices = range(EVENT_SIZE)
        masked_events = self.events[mask]
        
        # Plot individual events with device labels if available
        # Track which device labels we've already added to legend
        device_labels_plotted = set()
        for idx in masked_indices:
            event = self.events[idx]
            device_idx = self.devices[idx] if idx < len(self.devices) else -1
            
            # Label events if they are matched to a device
            if device_idx != -1:
                device_label = self.device_labels[device_idx]
                if device_label and device_label not in device_labels_plotted:
                    # First time seeing this device label - add to legend
                    ax.plot(power_indices, event, linewidth=1.0, alpha=0.3, color='blue', label=device_label)
                    device_labels_plotted.add(device_label)
                elif device_label:
                    # Already seen this device label - don't add to legend again
                    ax.plot(power_indices, event, linewidth=1.0, alpha=0.3, color='blue')
                else:
                    ax.plot(power_indices, event, linewidth=0.5, alpha=0.3, color='gray')
            else:
                ax.plot(power_indices, event, linewidth=0.5, alpha=0.3, color='gray')
        
        if device_pattern is not None:
            ax.plot(power_indices, device_pattern, linewidth=3, color='green', label='device')
        median_pattern = np.median(masked_events, axis=0)
        ax.plot(power_indices, median_pattern, linewidth=2, color='yellow', label='Median')
        # Set title and labels
        if title_label:
            ax.set_title(f'Power Pattern {title_label}')
        else:
            ax.set_title('Power Pattern')
        ax.set_xlabel(f'Power over {EVENT_SIZE} samples')
        ax.set_ylabel('Power')
        ax.grid(True, alpha=0.3)
        # Add legend if there are device labels
        if len(device_labels_plotted) > 0:
            ax.legend()
        
        # Highlight specific event if requested
        if highlight_event_data is not None:
            ax.plot(power_indices, highlight_event_data, linewidth=2, color='red', label='Current Event', alpha=0.8)
            ax.legend()
        elif highlight_event_index is not None and highlight_event_index < len(self.events) and mask[highlight_event_index]:
            highlight_event = self.events[highlight_event_index]
            ax.plot(power_indices, highlight_event, linewidth=2, color='red', label='Current Event', alpha=0.8)
            ax.legend()
        
        plt.tight_layout()
        # Restore or set window position
        self.restore_window_position(fig, 'power')
        if plt.isinteractive():
            plt.show(block=False)
        return fig, ax
    
    def most_distinct(self, eligible_clusters):
        """Find the most distinctive cluster from eligible clusters"""
        # Calculate median distance from each cluster center to all other cluster centers
        # Use the cluster medians we already calculated
        distinctiveness_scores = {}
        for i in eligible_clusters:
            # Calculate median distance to all other cluster centers
            distances = []
            for j in range(self.n_clusters):
                if i != j:
                    distances.append(np.linalg.norm(self.cluster_medians[i] - self.cluster_medians[j]))
            distinctiveness_scores[i] = np.mean(distances)
    
        best_cluster_id = max(distinctiveness_scores, key=distinctiveness_scores.get)
        return best_cluster_id
    
    def browse_events(self, db=None):
        """Browse through events, showing each one with matching events based on distance"""
        if self.events is None or len(self.events) == 0:
            print("❌ No events available to browse.")
            return
        
        # Start with the most recent event (last index)
        current_index = len(self.events) - 1
        # Default match distance threshold: smallest of max_device_distances, or 500 if no devices
        if len(self.max_device_distances) > 0:
            match_distance = min(self.max_device_distances)
        else:
            match_distance = 500
        fig_timeline = None
        fig_power = None
        fig_hist = None
        
        while True:
            # Get current event data
            current_event = self.events[current_index]
            current_cluster_id = self.clusters[current_index] if current_index < len(self.clusters) else -1
            current_device_idx = self.devices[current_index] if current_index < len(self.devices) else -1
            current_timestamp = self.timestamps[current_index]
            
            print("="*60)
            print(f"Event {current_index + 1} of {len(self.events)}")
            print(f"Timestamp: {current_timestamp}")
            print(f"Cluster ID: {current_cluster_id}")
            if current_device_idx != -1:
                device_label = self.device_labels[current_device_idx]
                print(f"Device: {device_label} (ID: {current_device_idx})")
            else:
                print("Device: Unassigned")
            
            # Find matching events based on distance
            # Scale the current event and all events
            current_event_scaled = myscaler(current_event.reshape(1, -1))
             
            # Calculate distances from current event to all other events
            distances = cdist(current_event_scaled, self.scaled_events, metric='euclidean')[0]
            
            # Find matches where distance is less than match_distance
            matching_mask = distances < match_distance
            matching_count = np.sum(matching_mask)
            print(f"Matching events (distance < {match_distance}): {matching_count}")
            
            # Show distance to current event (should be 0)
            if matching_count > 0:
                matching_distances = distances[matching_mask]
                min_dist = np.min(matching_distances)
                max_dist = np.max(matching_distances)
                print(f"  Distance range: {min_dist:.2f} to {max_dist:.2f}")
            
            # Plot timeline with matching events
            fig_timeline, _ = self.plot_timeline(matching_mask, title_label="All Events")
            # Plot the day of week histogram for matching events
            fig_hist, _ = self.plot_day_of_week_histogram(matching_mask, title_label=f"Matching Events (distance < {match_distance})")
            
            # Plot power pattern for matching events
            # Calculate median pattern from matching events and call plot_power
            if matching_count > 0:
                matching_events = self.events[matching_mask]
                median_pattern = np.median(matching_events, axis=0)
                title = f"Event {current_index + 1} - Matching Events (distance < {match_distance})"
                plot_mask = matching_mask
                # Highlight current event if it's in the matching set
                highlight_index = current_index if matching_mask[current_index] else None
            else:
                # If no matches, create a mask with just the current event
                plot_mask = np.zeros(len(self.events), dtype=bool)
                plot_mask[current_index] = True
                median_pattern = current_event
                title = f"Event {current_index + 1} - No Matches"
                highlight_index = current_index
            fig_power, _ = self.plot_power(plot_mask, title_label=title, device_pattern=median_pattern, 
                                                   highlight_event_index=highlight_index)
            
            # Ask for user action
            print("\n" + "="*60)
            print("BROWSE OPTIONS")
            print("="*60)
            print("N - Next event")
            print("P - Previous event")
            print(f"M - Change match distance (current: {match_distance})")
            print("S - Save matching events as a device")
            print("Q - Quit browse mode")
            print("="*60)
            try:
                action = input("Enter your choice (N, P, M, S, or Q): ").strip().upper()
            except KeyboardInterrupt:
                print("\nExiting browse mode...")
                break
            
            if action == 'Q':
                break
            elif action == 'N':
                if current_index < len(self.events) - 1:
                    current_index += 1
                else:
                    print("Already at the last event.")
            elif action == 'P':
                if current_index > 0:
                    current_index -= 1
                else:
                    print("Already at the first event.")
            elif action == 'S':
                # Save matching events as a device
                if matching_count == 0:
                    print("❌ No matching events to save as a device.")
                else:
                    try:
                        device_name = input("Enter device name: ").strip()
                        if device_name:
                            # Find next available device ID
                            if len(self.device_labels) > 0:
                                next_device_idx = len(self.device_labels)
                            else:
                                next_device_idx = 0
                            next_device_key = max(self.device_keys)+1 if len(self.device_keys) > 0 else 0
                            # Append to device arrays
                            self.device_keys = np.append(self.device_keys, next_device_key)
                            self.device_labels = np.append(self.device_labels, device_name)
                            self.device_profiles = np.vstack([self.device_profiles, median_pattern.reshape(1, -1)])
                            self.max_device_distances = np.append(self.max_device_distances, match_distance)
                            self.off_delays = np.append(self.off_delays, np.array([-1], dtype=int))
                            self.scaled_device_profiles = myscaler(self.device_profiles)
                            
                            self.db.add_device(next_device_key, device_name, match_distance, -1, median_pattern)
                            
                            print(f"✅ Saved as device '{device_name}' (ID: {next_device_idx})")
                            print(f"   Profile: median of {matching_count} matching events")
                            print(f"   Max distance: {match_distance:.2f}")
                            
                            # Reassign devices to update assignments
                            self._reassign_devices_from_medians()
                        else:
                            print("❌ Device name cannot be empty.")
                    except KeyboardInterrupt:
                        print("\nCancelled.")
            elif action == 'M':
                try:
                    new_distance_input = input(f"Enter new match distance (current: {match_distance}): ").strip()
                    if new_distance_input:
                        new_distance = float(new_distance_input)
                        if new_distance > 0:
                            match_distance = new_distance
                            print(f"✅ Match distance updated to {match_distance}")
                        else:
                            print("❌ Match distance must be greater than 0.")
                    else:
                        print("No change made.")
                except ValueError:
                    print("❌ Invalid value. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nCancelled.")
            else:
                print("Invalid input. Please enter N, P, E, M, or Q")
        
        # Close plots when exiting
        if fig_timeline is not None:
            self.save_window_position(fig_timeline, 'timeline')
            plt.close(fig_timeline)
        if fig_power is not None:
            self.save_window_position(fig_power, 'power')
            plt.close(fig_power)
        if fig_hist is not None:
            self.save_window_position(fig_hist, 'histogram')
            plt.close(fig_hist)
    
    def most_compact(self, eligible_clusters):
        """Find the most compact cluster from eligible clusters (smallest average distance from points to center)"""
        compactness_scores = {}
        for i in eligible_clusters:
            # Get all events in this cluster
            cluster_mask = self.clusters == i
            cluster_events = self.events[cluster_mask]
            # Calculate average distance from each point to the cluster center
            distances = []
            for event in cluster_events:
                distances.append(np.linalg.norm(event - self.cluster_medians[i]))
            compactness_scores[i] = np.mean(distances)
    
        # Most compact = smallest average distance
        best_cluster_id = min(compactness_scores, key=compactness_scores.get)
        return best_cluster_id
    
    def most_power(self, eligible_clusters):
        """Find the most power cluster from eligible clusters (largest average power)"""
        power_scores = {}
        for i in eligible_clusters:
            power_scores[i] = np.mean(np.abs(self.events[self.clusters == i]))
        best_cluster_id = max(power_scores, key=power_scores.get)
        return best_cluster_id
    
 
    def edit(self):
        print("="*60)
        print("Device Editor")
        print("="*60)
        
        if len(self.device_labels) == 0:
            print("❌ No devices found to edit.")
            return
        fig_timeline = None
        fig_power = None
        fig_hist = None
        # Loop until user presses CTRL-C
        while True:
            # Display all devices with their IDs and labels
            self.display_devices_report()
            # Ask user which device to edit/delete by device_id
            try:
                device_id_input = input("Enter the device ID to edit or delete (Q to exit): ").strip()
            except KeyboardInterrupt:
                print("\nCancelled.")
                break
            if device_id_input.upper() == 'Q':
                break
            # Convert to integer (device index)
            try:
                device_idx = int(device_id_input)
            except ValueError:
                print(f"❌ Invalid device ID: '{device_id_input}'. Please enter a number.")
                continue
            
            # Check if device exists (device_idx must be a valid array index)
            if device_idx < 0 or device_idx >= len(self.device_labels):
                print(f"❌ Device ID {device_idx} not found.")
                continue
            
            device_label = self.device_labels[device_idx]
            
            # Inner loop for editing this specific device
            while True:
                current_off_delay = self.off_delays[device_idx]
                current_max_distance = self.max_device_distances[device_idx]                
                mask = self.devices == device_idx
                device_label = self.device_labels[device_idx]
                # Calculate median pattern for the device
                device_pattern = self.device_profiles[device_idx]
                fig_timeline, _ = self.plot_timeline(mask, title_label=device_label)
                fig_hist, _ = self.plot_day_of_week_histogram(mask, title_label=device_label)
                fig_power, _ = self.plot_power(mask, title_label=device_label, device_pattern=device_pattern)
                # Ask what to do
                max_dist_str = f"{current_max_distance:.2f}" if current_max_distance is not None else "N/A"
                off_delay_str = f"{current_off_delay} min" if current_off_delay >= 0 else f"{current_off_delay}"
                print(f"\nSelected device: ID {device_idx}, Label: '{device_label}', off_delay: {off_delay_str}, max_distance: {max_dist_str}")
                print("="*60)
                print("EDIT OPTIONS")
                print("="*60)
                print("N - Edit device name")
                print("D - Edit off_delay (in minutes)")
                print("M - Edit max_device_distance")
                print("X - Delete device")
                print("Q - Quit editing this device")
                print("="*60)
                try:
                    action = input("Enter your choice (N, D, M, X, or Q): ").strip().upper()
                except KeyboardInterrupt:
                    print("\nCancelled.")
                    break  # Exit inner loop, return to device list
                
                if action == 'X':
                    # Get device_key before deletion
                    device_key = self.device_keys[device_idx] if device_idx < len(self.device_keys) else None
                    
                    # Delete from database first if device_key exists
                    if device_key is not None and self.db is not None:
                        self.db.delete_device(device_key)
                    
                    # Remove device from all arrays
                    mask = np.ones(len(self.device_labels), dtype=bool)
                    mask[device_idx] = False
                    self.device_labels = self.device_labels[mask]
                    self.device_profiles = self.device_profiles[mask]
                    self.max_device_distances = self.max_device_distances[mask]
                    self.off_delays = self.off_delays[mask]
                    # Also remove from device_keys array
                    if device_idx < len(self.device_keys):
                        self.device_keys = self.device_keys[mask]
                    self.scaled_device_profiles = myscaler(self.device_profiles)
                    
                    # Recalculate device assignments after deletion
                    if self.events is not None and len(self.events) > 0:
                        self._reassign_devices_from_medians()
                    
                    print(f"✅ Device '{device_label}' (ID {device_idx}) deleted successfully.")
                    # Close plots when deleting device
                    break  # Exit inner loop, return to device list
                elif action == 'N':
                    # Ask for new name
                    try:
                        new_label = input(f"Enter new label for '{device_label}' (CTRL-C to cancel): ").strip()
                    except KeyboardInterrupt:
                        print("\nCancelled.")
                        continue  # Go back to edit options for this device
                    if not new_label:
                        print("❌ Label cannot be empty.")
                        continue
                    # Update the label
                    self.device_labels[device_idx] = new_label
                    device_label = new_label  # Update local variable
                    print(f"✅ Device renamed from '{device_label}' to '{new_label}'.")
                    # Update device in database
                    device_key = self.device_keys[device_idx]
                    self.db.modify_device(device_key, device_label=new_label)
                elif action == 'D':
                    # Ask for new off_delay
                    try:
                        current_off_delay_str = f"{current_off_delay} min" if current_off_delay >= 0 else f"{current_off_delay}"
                        off_delay_input = input(f"Enter new off_delay in minutes (current: {current_off_delay_str}, CTRL-C to cancel): ").strip()
                        if off_delay_input:
                            new_off_delay = int(off_delay_input)  # Convert to int
                            self.off_delays[device_idx] = new_off_delay
                            new_off_delay_str = f"{new_off_delay} min" if new_off_delay >= 0 else f"{new_off_delay}"
                            print(f"✅ Device off_delay updated from {current_off_delay_str} to {new_off_delay_str}.")
                            # Update device in database
                            device_key = self.device_keys[device_idx]
                            self.db.modify_device(device_key, off_delay=new_off_delay)
                        else:
                            print("No change made.")
                    except ValueError:
                        print("❌ Invalid value. Please enter a number.")
                    except KeyboardInterrupt:
                        print("\nCancelled.")
                        continue  # Go back to edit options for this device
                elif action == 'M':
                    # Ask for new max_device_distance
                    try:
                        max_dist_input = input(f"Enter new max_device_distance (current: {max_dist_str}, CTRL-C to cancel): ").strip()
                        if max_dist_input:
                            new_max_distance = float(max_dist_input)
                            self.max_device_distances[device_idx] = new_max_distance
                            print(f"✅ Device max_distance updated from {max_dist_str} to {new_max_distance:.2f}.")
                            # Recalculate device assignments with new max_distance
                            if self.events is not None and len(self.events) > 0:
                                print("🔄 Recalculating device assignments...")
                                self._reassign_devices_from_medians()
                            # Update device in database
                            device_key = self.device_keys[device_idx]   
                            self.db.modify_device(device_key, max_device_distance=new_max_distance)
                        else:
                            print("No change made.")
                    except ValueError:
                        print("❌ Invalid value. Please enter a number.")
                    except KeyboardInterrupt:
                        print("\nCancelled.")
                        continue  # Go back to edit options for this device
                elif action == 'Q':
                    break  # Exit inner loop, return to device list
                else:
                    print("Invalid input. Please enter N, D, M, X, or Q")
        # Close plots when exiting
        if fig_timeline is not None:
            self.save_window_position(fig_timeline, 'timeline')
            plt.close(fig_timeline)
        if fig_power is not None:
            self.save_window_position(fig_power, 'power')
            plt.close(fig_power)
        if fig_hist is not None:
            self.save_window_position(fig_hist, 'histogram')
            plt.close(fig_hist)

def get_user_choice():
    """
    Ask user whether to run new analysis or load existing results
    """
    print("="*60)
    print("🔍 DATA ANALYSIS OPTIONS")
    print("="*60)
    print("F - Fresh analysis (recommended for new data)")
    print("L - Load previous analysis results")
    print("R - Redo clustering on unlabeled events")
    print("I - Identify clusters as devices")
    print("P - Plot all events")
    print("E - Edit existing devices")
    print("B - Browse events")
    print("X - Export devices to CSV file")
    print("M - Import devices from CSV file")
    print("Q - Quit")
    print("="*60)
    while True:
        try:
            choice = input("Enter your choice (F, L, R, I, P, E, B, X, M, or Q): ").strip().upper()
            if choice == 'F':
                return "fresh"
            elif choice == 'L':
                return "load"
            elif choice == 'R':
                return "remove"
            elif choice == 'I':
                return "label"
            elif choice == 'P':
                return "plot"
            elif choice == 'E':
                return "edit"
            elif choice == 'B':
                return "browse"
            elif choice == 'X':
                return "export"
            elif choice == 'M':
                return "import"
            elif choice == 'Q':
                return "quit"
            else:
                print("Please enter F, L, R, I, P, E, B, X, M, or Q")
        except KeyboardInterrupt:
            print("\nExiting...")
            return "quit"
        except:
            print("Invalid input. Please enter F, L, R, I, P, E, B, X, M, or Q")


def interactive_cluster_labeling(analyzer, db=None):
    """
    Interactive cluster labeling function - loops through all clusters
    When a cluster is labeled, it becomes a device.
    
    Args:
        analyzer: PowerEventAnalyzer instance
        db: Optional PowerEventDatabase instance to add devices to database
    """
    print("="*60)
    print("🏷️  IDENTIFY CLUSTERS AS DEVICES")
    print("="*60)
    print("This will loop through all clusters and allow you to label them.")
    print("When a cluster is labeled, it becomes a device.")
    print("="*60)
    
    # Get all clusters
    all_clusters = sorted(range(analyzer.n_clusters))
    fig_timeline = None
    fig_power = None
    fig_hist = None
    
    for cluster_id in all_clusters:
        # Show cluster pattern
        print("="*60)
        print(f"📊 Cluster {cluster_id} of {analyzer.n_clusters - 1}")
        print("="*60)
        
        mask = analyzer.clusters == cluster_id
        cluster_label = analyzer._get_device_label_for_cluster(cluster_id)
        if cluster_label is None:
            cluster_label = f"Cluster {cluster_id}"
        # Calculate median pattern for the cluster
        median_pattern = analyzer.cluster_medians[cluster_id]
        fig_timeline, ax_timeline = analyzer.plot_timeline(mask, title_label=cluster_label)
        fig_hist, ax_hist = analyzer.plot_day_of_week_histogram(mask, title_label=cluster_label)
        fig_power, ax_power = analyzer.plot_power(mask, title_label=cluster_label, device_pattern=median_pattern)
        
        current_label = analyzer._get_device_label_for_cluster(cluster_id)
        cluster_count = np.sum(analyzer.clusters == cluster_id)
        print(f"Cluster {cluster_id}: {cluster_count} events")
        print(f"Current device label: '{current_label}'")
        print("="*60)
        print("LABELING OPTIONS")
        print("="*60)
        print("Enter label - Set label for this cluster (becomes a device)")
        print("S - Skip this cluster")
        print("Q - Quit labeling")
        print("="*60)
        
        try:
            user_input = input(f"Enter label for cluster {cluster_id} (or S/Q): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if user_input.upper() == 'Q':
            print("Quitting cluster labeling...")
            break
        elif user_input.upper() == 'S':
            print(f"⏭️  Skipped cluster {cluster_id}")
            continue
        elif user_input:
            # Convert labeled cluster to device immediately
            device_idx = analyzer._convert_cluster_to_device(cluster_id, user_input)
            print(f"✅ Cluster {cluster_id} labeled as '{user_input}' (now device {device_idx})")
        else:
            print("❌ Label cannot be empty. Use S to skip or Q to quit.")
    
    # Close plots
    if fig_timeline is not None:
        analyzer.save_window_position(fig_timeline, 'timeline')
        plt.close(fig_timeline)
    if fig_power is not None:
        analyzer.save_window_position(fig_power, 'power')
        plt.close(fig_power)
    if fig_hist is not None:
        analyzer.save_window_position(fig_hist, 'histogram')
        plt.close(fig_hist)
    # need to reassign devices from medians for the new devices 
    analyzer._reassign_devices_from_medians()



if __name__ == "__main__":
    # Configure logging - only show messages from powerMonitor and poweranalyzer
    logging.basicConfig(
        level=logging.WARNING,  # Suppress INFO/DEBUG from third-party modules
        format="%(message)s"
    )
    # Set specific loggers to INFO level
    logging.getLogger('powerMonitor').setLevel(logging.INFO)
    logging.getLogger('poweranalyzer').setLevel(logging.INFO)
    # Also set for when run as script (__main__)
    if __name__ == '__main__':
        logging.getLogger('__main__').setLevel(logging.INFO)

    
    print("🚀 Power Event Analysis with Smart Data Detection")
    print("=" * 60)
    analyzer = None
    bigfig = None
    analyzer = PowerEventAnalyzer()
    analyzer.load_events()# Load all data files
    status = analyzer.get_status()
    if analyzer.n_events > 0:
        print(f"📊 Loaded: {analyzer.n_events} events from database")
        print(f"   Earliest event: {analyzer.first_event.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Latest event: {analyzer.last_event.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Most recent analysis: {analyzer.date_of_analysis_in_memory.strftime('%Y-%m-%d %H:%M:%S') if analyzer.date_of_analysis_in_memory is not None else 'No analysis date found'}")
    while True:
        choice = get_user_choice()  
        plt.close(bigfig)
        if choice == "fresh":
            print("\n🔄 Running fresh analysis...")
            analyzer.perform_clustering()    
            print(f"Identified {analyzer.n_clusters} clusters")
            print("\nCluster summary:")
            print(analyzer.get_cluster_summary())
    
        elif choice == "load":
            print("\n📂 Loading previous analysis results...")
            analyzer.load_results_from_database()
        elif choice == "remove":
            print("\n🗑️  Performing clustering on unlabeled events...")
            analyzer.perform_clustering(unlabeled=True)
            print(f"Identified {analyzer.n_clusters} clusters")
            print("\nCluster summary:")
            print(analyzer.get_cluster_summary())
        elif choice == "label":
            print("\n🏷️  Starting cluster identification...")
            interactive_cluster_labeling(analyzer)
        elif choice == "plot":
            print(f"Found {analyzer.n_events} events")
            print(f"Identified {analyzer.n_clusters} clusters")
            print("\nCluster summary:")
            print(analyzer.get_cluster_summary())
            # Create timeline plot
            bigfig, bigax = analyzer.plot_timeline(mask=None, title_label="All Events")
        elif choice == "edit":
            print("\n✏️  Editing existing devices...")
            if analyzer is None:
                print("❌ No analysis available. Please run analysis first (choice 1 or 2).")
            else:
                analyzer.edit()
        elif choice == "browse":
            print("\n🔍 Browsing events...")
            analyzer.browse_events()
        elif choice == "export":
            print("\n📤 Exporting devices to CSV file...")
            try:
                filename = input("Enter filename (or press Enter for './data/devices_export.csv'): ").strip()
                if not filename:
                    filename = './data/devices_export.csv'
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
                
                success = analyzer.db.export_devices(filename)
                if success:
                    print(f"✅ Devices exported successfully to {filename}")
                else:
                    print(f"❌ Failed to export devices")
            except Exception as e:
                logger.error(f"Error exporting devices: {e}")
                print(f"❌ Error exporting devices: {e}")
        elif choice == "import":
            print("\n📥 Importing devices from CSV file...")
            try:
                filename = input("Enter filename (or press Enter for './data/devices_export.csv'): ").strip()
                if not filename:
                    filename = './data/devices_export.csv'
                
                if not os.path.exists(filename):
                    print(f"❌ File not found: {filename}")
                    continue
                
                # Confirm replacement
                confirm = input("⚠️  This will replace all existing devices. Continue? (yes/no): ").strip().lower()
                if confirm not in ['yes', 'y']:
                    print("Import cancelled.")
                    continue
                
                success = analyzer.db.import_devices(filename)
                if success:
                    print(f"✅ Devices imported successfully from {filename}")
                    # Reload devices into analyzer if it exists
                    if analyzer is not None:
                        print("🔄 Reloading devices into analyzer...")
                        _ = analyzer.load_results_from_database()
                        print("✅ Devices reloaded into analyzer")
                else:
                    print(f"❌ Failed to import devices")
            except Exception as e:
                logger.error(f"Error importing devices: {e}")
                print(f"❌ Error importing devices: {e}")
        elif choice == "quit":
            break

    print("\n✅ Analysis session complete!")
    