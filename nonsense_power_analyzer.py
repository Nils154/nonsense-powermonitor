#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonsense Power Analyzer - Web Application
A web-based interface for power event analysis, clustering, and device identification.

Based on poweranalyzer.py
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import time
from datetime import datetime
import logging
import base64
import io
from dateutil import parser
from poweranalyzer import PowerEventAnalyzer, myscaler
from database import PowerEventDatabase
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
        
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Global analyzer instance (in production, use proper session management or database)
current_event_index = None
current_match_distance = 500
analyzer = None
mode = 'monitor'

# Progress tracking for plot generation
plot_progress_scaling_factor = 0.5  # Calibrates over time


@app.route('/')
def index():
    """Main dashboard"""
    global current_event_index, current_match_distance, analyzer, mode
    # Get status from analyzer
    status = analyzer.get_status() if analyzer is not None else {
        'latest_analysis': None,
        'number_of_events': 0,
        'last_event': None,
        'first_event': None,
        'n_devices': 0
    }
    
    # Send first_event as ISO format so frontend can convert to local timezone
    if status['first_event'] is not None:
        if status['first_event'].tzinfo is not None:
            status['first_event'] = status['first_event'].isoformat()
        else:
            # If naive, assume it's in local timezone and add timezone info
            from datetime import datetime
            local_tz = datetime.now().astimezone().tzinfo
            status['first_event'] = status['first_event'].replace(tzinfo=local_tz).isoformat()
    
    # create the devices table
    devices_list = []
    if analyzer is not None and analyzer.n_devices > 0:
         for device_idx in range(analyzer.n_devices):
            label = analyzer.device_labels[device_idx]
            off_delay = analyzer.off_delays[device_idx]
            max_distance = analyzer.max_device_distances[device_idx] 
            # Calculate ON/OFF and average power
            on_off = "N/A"
            avg_power = "N/A"
            profile = analyzer.device_profiles[device_idx]
            avg_power = float(np.mean(profile))
            on_off = "ON" if avg_power > 0 else "OFF" if avg_power <= 0 else "N/A"
            count = int(np.sum(analyzer.devices == device_idx))
            devices_list.append({
                'idx': device_idx,
                'label': label,
                'on_off': on_off,
                'avg_power': round(avg_power, 2) if avg_power != "N/A" else None,
                'count': count,
                'max_distance': round(max_distance, 2) if max_distance is not None else None,
                'off_delay': off_delay if off_delay != -1 else None
            })
    return render_template('index.html', 
                         status=status,
                         devices=devices_list)


@app.route('/api/fresh_analysis', methods=['POST'])
def api_fresh_analysis():
    """Run fresh clustering analysis"""
    global analyzer, current_event_index, current_match_distance, mode
    
    if analyzer is None or analyzer.n_events == 0:
        return jsonify({'success': False, 'message': 'No analysis available'})
    
    mode = 'labeling'
    logger.info("⏸️  Pausing data reloading for cluster labeling")
    analyzer.perform_clustering()
    
    return jsonify({
        'success': True,
        'message': f'Analysis complete: {analyzer.n_clusters} clusters found',
        'n_clusters': analyzer.n_clusters,
        'n_events': analyzer.n_events
    })

@app.route('/api/analyze_unlabeled', methods=['POST'])
def api_analyze_unlabeled():
    """Analyze unlabeled events by removing labeled events and re-clustering"""
    global analyzer, timestamps, events, is_labeling_mode
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis available. Please run analysis first.'})
    
    if timestamps is None or events is None:
        return jsonify({'success': False, 'message': 'No data available'})
    
    try:
        # Set labeling mode flag to pause data reloading
        is_labeling_mode = True
        logger.info("⏸️  Pausing data reloading for cluster labeling")
        # Perform clustering on unlabeled events
        analyzer.perform_clustering(unlabeled=True)
        
        logger.info(f"Found {len(analyzer.timestamps)} unlabeled events")
        logger.info(f"Identified {analyzer.n_clusters} new clusters")
        
        return jsonify({
            'success': True,
            'message': f'Analysis complete: {analyzer.n_clusters} clusters found from unlabeled events',
            'n_clusters': analyzer.n_clusters,
            'n_events': len(analyzer.timestamps) if analyzer.timestamps is not None else 0
        })
    except Exception as e:
        logger.error(f"Error analyzing unlabeled events: {e}")
        import traceback
        traceback.print_exc()
        # Reset flag on error
        is_labeling_mode = False
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/get_cluster_info', methods=['GET'])
def api_get_cluster_info():
    """Get information about a specific cluster"""
    global analyzer
    
    if analyzer is None or analyzer.events is None or len(analyzer.events) == 0:
        return jsonify({'success': False, 'message': 'No analysis available'})
    
    cluster_id = request.args.get('cluster_id', type=int)
    if cluster_id is None or cluster_id < 0 or cluster_id >= analyzer.n_clusters:
        return jsonify({'success': False, 'message': 'Invalid cluster ID'})
    
    try:
        mask = analyzer.clusters == cluster_id
        cluster_count = np.sum(mask)
        cluster_label = analyzer._get_device_label_for_cluster(cluster_id)
        if cluster_label is None:
            cluster_label = f"Cluster {cluster_id}"
        
        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_count': int(cluster_count),
            'n_clusters': analyzer.n_clusters
        })
    except Exception as e:
        logger.error(f"Error getting cluster info: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_cluster_plots', methods=['GET'])
def api_get_cluster_plots():
    """Get plots for a specific cluster"""
    global analyzer
    
    if analyzer is None or analyzer.events is None or len(analyzer.events) == 0:
        return jsonify({'success': False, 'message': 'No analysis available'})
    
    cluster_id = request.args.get('cluster_id', type=int)
    if cluster_id is None or cluster_id < 0 or cluster_id >= analyzer.n_clusters:
        return jsonify({'success': False, 'message': 'Invalid cluster ID'})
    
    try:
        mask = analyzer.clusters == cluster_id
        cluster_count = np.sum(mask)
        
        if cluster_count == 0:
            return jsonify({'success': False, 'message': f'No events found for cluster {cluster_id}'})
        
        cluster_label = analyzer._get_device_label_for_cluster(cluster_id)
        if cluster_label is None:
            cluster_label = f"Cluster {cluster_id}"
        
        # Calculate median pattern for the cluster
        median_pattern = analyzer.cluster_medians[cluster_id]
        
        # Generate plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Timeline plot
        timeline_title = f"{cluster_label} ({cluster_count} events)"
        fig_timeline, ax_timeline = analyzer.plot_timeline(mask, title_label=timeline_title)
        timeline_img = io.BytesIO()
        fig_timeline.savefig(timeline_img, format='png', bbox_inches='tight', dpi=100)
        timeline_img.seek(0)
        timeline_base64 = base64.b64encode(timeline_img.getvalue()).decode('utf-8')
        plt.close(fig_timeline)
        
        # Power plot
        power_title = f"{cluster_label} - {cluster_count} events"
        fig_power, ax_power = analyzer.plot_power(mask, title_label=power_title,
                                                   device_pattern=median_pattern,
                                                   highlight_event_index=None)
        power_img = io.BytesIO()
        fig_power.savefig(power_img, format='png', bbox_inches='tight', dpi=100)
        power_img.seek(0)
        power_base64 = base64.b64encode(power_img.getvalue()).decode('utf-8')
        plt.close(fig_power)
        
        # Histogram plot
        hist_title = f"{cluster_label} ({cluster_count} events)"
        fig_hist, ax_hist = analyzer.plot_day_of_week_histogram(mask, title_label=hist_title)
        hist_img = io.BytesIO()
        fig_hist.savefig(hist_img, format='png', bbox_inches='tight', dpi=100)
        hist_img.seek(0)
        hist_base64 = base64.b64encode(hist_img.getvalue()).decode('utf-8')
        plt.close(fig_hist)
        
        return jsonify({
            'success': True,
            'timeline': timeline_base64,
            'power': power_base64,
            'histogram': hist_base64,
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_count': int(cluster_count)
        })
    except Exception as e:
        logger.error(f"Error generating cluster plots: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/label_cluster', methods=['POST'])
def api_label_cluster():
    """Label a cluster as a device"""
    global analyzer
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis available'})
    
    data = request.get_json()
    cluster_id = data.get('cluster_id')
    label = data.get('label')
    
    if cluster_id is None:
        return jsonify({'success': False, 'message': 'Cluster ID required'})
    
    if not label or not label.strip():
        return jsonify({'success': False, 'message': 'Label cannot be empty'})
    
    try:
        device_idx = analyzer._convert_cluster_to_device(cluster_id, label.strip())
        return jsonify({
            'success': True,
            'message': f'Cluster {cluster_id} labeled as "{label}" (now device {device_idx})',
            'device_idx': device_idx
        })
    except Exception as e:
        logger.error(f"Error labeling cluster: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/find_event_by_timestamp', methods=['GET'])
def api_find_event_by_timestamp():
    """Find the nearest event to a given timestamp"""
    global analyzer, current_event_index, mode
    
    if analyzer is None or analyzer.events is None or len(analyzer.events) == 0:
        return jsonify({'success': False, 'message': 'No analysis or data available'})
    
    timestamp_str = request.args.get('timestamp', '')
    if not timestamp_str:
        return jsonify({'success': False, 'message': 'Timestamp required'})
    
    try:
        # Try to parse the timestamp string
        # Support various formats like "Jan 5, 4:03am", "2024-01-05 04:03", etc.
        try:
            # Use fuzzy parsing to handle various formats
            target_timestamp = parser.parse(timestamp_str, fuzzy=True)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Could not parse timestamp "{timestamp_str}": {str(e)}'})
        
        # Ensure target_timestamp is timezone-aware (add local timezone if naive)
        if target_timestamp.tzinfo is None:
            from datetime import datetime
            local_tz = datetime.now().astimezone().tzinfo
            target_timestamp = target_timestamp.replace(tzinfo=local_tz)
        
        # Find the event with the closest timestamp
        timestamps_array = analyzer.timestamps
        if timestamps_array is None or len(timestamps_array) == 0:
            return jsonify({'success': False, 'message': 'No timestamps available'})
        
        # Convert timestamps_array to numpy array if needed
        timestamps_array = np.asarray(timestamps_array, dtype=object)
        
        # Ensure all timestamps in array are timezone-aware for comparison
        # Get local timezone once
        from datetime import datetime
        local_tz = datetime.now().astimezone().tzinfo
        
        # Calculate absolute differences using numpy
        # Convert to timedelta objects and get total seconds
        differences_seconds = np.array([
            abs(((ts if ts.tzinfo is not None else ts.replace(tzinfo=local_tz)) - target_timestamp).total_seconds())
            for ts in timestamps_array
        ])
        
        # Find the index with minimum difference
        nearest_index = int(np.argmin(differences_seconds))
        
        # Update server-side current_event_index and mode
        current_event_index = nearest_index
        mode = 'browse'  # User manually navigated, switch to browse mode
        
        # Ensure index is valid
        if current_event_index >= analyzer.n_events:
            current_event_index = analyzer.n_events - 1
        if current_event_index < 0:
            current_event_index = 0
        
        # Get the actual timestamp of the nearest event
        nearest_timestamp = timestamps_array[nearest_index]
        diff_seconds = float(differences_seconds[nearest_index])
        
        # Format timestamp for response
        if hasattr(nearest_timestamp, 'isoformat'):
            timestamp_str = nearest_timestamp.isoformat()
        else:
            timestamp_str = str(nearest_timestamp)
        
        # Format target timestamp for response
        if hasattr(target_timestamp, 'isoformat'):
            target_timestamp_str = target_timestamp.isoformat()
        else:
            target_timestamp_str = str(target_timestamp)
        
        return jsonify({
            'success': True,
            'event_index': int(nearest_index),
            'timestamp': timestamp_str,
            'target_timestamp': target_timestamp_str,
            'difference_seconds': float(diff_seconds)
        })
    except Exception as e:
        logger.error(f"Error finding event by timestamp: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/complete_labeling', methods=['POST'])
def api_complete_labeling():
    """Save results and complete cluster labeling"""
    global analyzer, mode
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis available'})
    
    analyzer.save_results_to_database()
    mode = 'monitor'
    logger.info("▶️  Resuming data reloading after cluster labeling")
    return jsonify({
        'success': True,
        'message': 'Labeling complete. Results saved.',
        'n_devices': analyzer.n_devices
    })

@app.route('/api/devices', methods=['GET'])
def api_get_devices():
    """Get list of all devices"""
    global analyzer
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis loaded'})
    
    devices = []
    if len(analyzer.device_labels) > 0:
        # device_labels is now a numpy array, iterate by index
        for device_idx in range(len(analyzer.device_labels)):
            label = analyzer.device_labels[device_idx]
            off_delay = analyzer.off_delays[device_idx] if device_idx < len(analyzer.off_delays) else -1
            max_distance = analyzer.max_device_distances[device_idx] if device_idx < len(analyzer.max_device_distances) else None
            
            # Calculate ON/OFF and average power
            on_off = "N/A"
            avg_power = "N/A"
            if device_idx < len(analyzer.device_profiles):
                profile = analyzer.device_profiles[device_idx]
                if len(profile) > 0:
                    avg_power = float(np.mean(profile))
                    on_off = "ON" if avg_power > 0 else "OFF" if avg_power <= 0 else "N/A"
                    
            
            count = int(np.sum(analyzer.devices == device_idx)) if hasattr(analyzer, 'devices') and analyzer.devices is not None else 0
            
            devices.append({
                'idx': device_idx,
                'label': label,
                'on_off': on_off,
                'avg_power': round(avg_power, 2) if avg_power != "N/A" else None,
                'count': count,
                'max_distance': round(max_distance, 2) if max_distance is not None else None,
                'off_delay': off_delay if off_delay != -1 else None
            })
    
    return jsonify({'success': True, 'devices': devices})

@app.route('/api/clusters', methods=['GET'])
def api_get_clusters():
    """Get list of all clusters"""
    global analyzer
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis loaded'})
    
    clusters = []
    for cluster_id in range(analyzer.n_clusters):
        cluster_mask = analyzer.clusters == cluster_id
        count = int(np.sum(cluster_mask))
        max_distance = analyzer.max_cluster_distances[cluster_id]
        device_label = analyzer._get_device_label_for_cluster(cluster_id)
        
        clusters.append({
            'id': cluster_id,
            'count': count,
            'max_distance': round(max_distance, 2) if max_distance is not None else None,
            'device_label': device_label
        })
    
    return jsonify({'success': True, 'clusters': clusters})

@app.route('/api/save_device', methods=['POST'])
def api_save_device():
    """Save a new device from matching events"""
    global analyzer, current_event_index, current_match_distance, mode
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis loaded'})
    
    data = request.json
    device_name = data.get('name')
    
    if not device_name:
        return jsonify({'success': False, 'message': 'Device name required'})
    
    try:
        # If event_index is provided, use the new add_device method (browse mode)
        if 'event_index' in data:
            event_index = int(data.get('event_index'))
            match_distance = float(data.get('match_distance', current_match_distance))
            
            if event_index < 0 or event_index >= analyzer.n_events:
                return jsonify({'success': False, 'message': f'Invalid event_index: {event_index}'})
            
            if match_distance <= 0:
                return jsonify({'success': False, 'message': 'Match distance must be greater than 0'})
            
            # Use the new add_device method
            device_idx = analyzer.add_device(event_index, match_distance, device_name)
            
            return jsonify({
                'success': True,
                'message': f'Device "{device_name}" saved from event {event_index}',
                'device_idx': device_idx
            })
        else:
            # Legacy method: use provided pattern and distance
            median_pattern = np.array(data.get('pattern'))
            match_distance = float(data.get('distance'))
            
            next_device_idx = len(analyzer.device_labels)
            
            # Add device to arrays (device_keys will be updated when saving to database)
            analyzer.device_labels = np.append(analyzer.device_labels, device_name)
            analyzer.device_profiles = np.vstack([analyzer.device_profiles, median_pattern.reshape(1, -1)])
            analyzer.max_device_distances = np.append(analyzer.max_device_distances, match_distance)
            analyzer.off_delays = np.append(analyzer.off_delays, np.array([-1], dtype=int))
            analyzer.scaled_device_profiles = myscaler(analyzer.device_profiles)
            
            # Reassign devices
            analyzer._reassign_devices_from_medians()
            
            # Save results to database
            analyzer.save_results_to_database()
            
            return jsonify({
                'success': True,
                'message': f'Device "{device_name}" saved',
                'device_idx': next_device_idx
            })
    except Exception as e:
        logger.error(f"Error saving device: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/check_updates', methods=['GET'])
def api_check_updates():
    """Check if data has been updated (for polling)"""
    global analyzer, current_event_index, current_match_distance, mode
    
    # Get status from database
    if mode == 'monitor':
        analyzer.get_status()

    if mode == 'monitor' and analyzer.n_events > current_event_index+1:
        # new events have been added to the database and we need to reload the events
        # and show the last event
        current_event_index = analyzer.n_events - 1
        #pull the last event from  the database
        analyzer.load_events()
        has_update = True
        logger.info(f"📊 Polling detected update: current event index changed to {current_event_index}")
    else:
        has_update = False
    # Send first_event_timestamp as ISO format so frontend can convert to local timezone
    first_event_iso = None
    if analyzer.first_event is not None:
        if analyzer.first_event.tzinfo is not None:
            first_event_iso = analyzer.first_event.isoformat()
        else:
            # If naive, assume it's in local timezone and add timezone info
            from datetime import datetime
            local_tz = datetime.now().astimezone().tzinfo
            first_event_iso = analyzer.first_event.replace(tzinfo=local_tz).isoformat()
    
    return jsonify({
        'success': True,
        'has_update': has_update,
        'event_count': analyzer.n_events,
        'device_count': analyzer.n_devices,
        'first_event_timestamp': first_event_iso
    })

@app.route('/api/get_plots', methods=['GET'])
def api_get_plots():
    """Get plots for current event as base64 images"""
    global analyzer, current_event_index, current_match_distance, mode
    global plot_progress_scaling_factor, plot_generation_times
    
    start_time = time.time()
    
    if analyzer is None or analyzer.events is None or analyzer.n_events == 0:
        return jsonify({'success': False, 'message': 'No analysis or data available'})
    
    # Get event_index from query parameter if provided
    requested_index = request.args.get('event_index', type=int)
    if requested_index is not None:
        current_event_index = requested_index
        mode = 'browse'  # User is navigating, switch to browse mode
    
    # Get match_distance from query parameter if provided (overrides device's max_distance)
    requested_match_distance = request.args.get('match_distance', type=float)
    if requested_match_distance is not None:
        current_match_distance = requested_match_distance
        mode = 'browse'  # User changed match distance, switch to browse mode
    
    if current_event_index is None:
        current_event_index = analyzer.n_events - 1
    if current_event_index >= analyzer.n_events:
        current_event_index = analyzer.n_events - 1
    if current_event_index < 0:
        current_event_index = 0 
    current_event = analyzer.events[current_event_index]
    current_device_idx = analyzer.devices[current_event_index] if current_event_index < len(analyzer.devices) else -1
    
    # Determine effective match distance
    # If match_distance was explicitly provided in query, use it (user override)
    # Otherwise, use device's max_distance if event matches a device, or global match_distance
    if requested_match_distance is not None:
        effective_match_distance = requested_match_distance
    elif current_device_idx != -1 and current_device_idx < len(analyzer.max_device_distances):
        # Use the device's max_device_distance for this event
        effective_match_distance = analyzer.max_device_distances[current_device_idx]
    else:
        # Use the global match_distance
        effective_match_distance = current_match_distance
        
    # Find matching events based on distance
    current_event_scaled = myscaler(current_event.reshape(1, -1))
    distances = cdist(current_event_scaled, analyzer.scaled_events, metric='euclidean')[0]
    matching_mask = distances < effective_match_distance
    matching_count = np.sum(matching_mask)
    
    # Calculate estimated progress based on matches found
    # Progress = (matching_count / total_events) * scaling_factor, capped at reasonable max
    estimated_progress = min(95, (matching_count * plot_progress_scaling_factor))
    
    # Calculate median pattern
    if matching_count > 0:
        matching_events = analyzer.events[matching_mask]
        median_pattern = np.median(matching_events, axis=0)
        highlight_index = current_event_index if matching_mask[current_event_index] else None
    else:
        matching_mask = np.zeros(len(analyzer.events), dtype=bool)
        matching_mask[current_event_index] = True
        median_pattern = current_event
        highlight_index = current_event_index
    
    # Timeline plot
    timeline_title = f"Events Matching Event Idx: {current_event_index})"
    fig_timeline, _ = analyzer.plot_timeline(matching_mask, title_label=timeline_title)
    timeline_img = io.BytesIO()
    fig_timeline.savefig(timeline_img, format='png', bbox_inches='tight', dpi=100)
    timeline_img.seek(0)
    timeline_base64 = base64.b64encode(timeline_img.getvalue()).decode('utf-8')
    plt.close(fig_timeline)
    
    # Power plot
    title = f"of Event Idx {current_event_index} and {matching_count} matches"
    fig_power, _ = analyzer.plot_power(matching_mask, title_label=title, 
                                                device_pattern=median_pattern,
                                                highlight_event_index=highlight_index)
    power_img = io.BytesIO()
    fig_power.savefig(power_img, format='png', bbox_inches='tight', dpi=100)
    power_img.seek(0)
    power_base64 = base64.b64encode(power_img.getvalue()).decode('utf-8')
    plt.close(fig_power)
    
    # Histogram plot
    hist_title = f"Matching Events Idx {current_event_index})"
    fig_hist, _ = analyzer.plot_day_of_week_histogram(matching_mask, 
                                                                title_label=hist_title)
    hist_img = io.BytesIO()
    fig_hist.savefig(hist_img, format='png', bbox_inches='tight', dpi=100)
    hist_img.seek(0)
    hist_base64 = base64.b64encode(hist_img.getvalue()).decode('utf-8')
    plt.close(fig_hist)
    
    # Get event info
    current_timestamp = analyzer.timestamps[current_event_index]
    device_label = None
    if current_device_idx != -1 and current_device_idx < len(analyzer.device_labels):
        device_label = analyzer.device_labels[current_device_idx]
    
    # Calculate actual generation time and calibrate scaling factor
    actual_time = time.time() - start_time
    time_per_matching_count = actual_time / matching_count
    plot_progress_scaling_factor = 0.9*plot_progress_scaling_factor + 0.1*time_per_matching_count
    logger.info(f"Plot progress scaling {plot_progress_scaling_factor:.3f} (time per matching count: {time_per_matching_count:.3f}s)")
    
    
    return jsonify({
        'success': True,
        'timeline': timeline_base64,
        'power': power_base64,
        'histogram': hist_base64,
        'event_index': current_event_index,
        'total_events': analyzer.n_events,
        'timestamp': str(current_timestamp),
        'device_idx': int(current_device_idx) if current_device_idx is not None else None,
        'device_label': device_label,
        'matching_count': int(matching_count),
        'match_distance': current_match_distance,  # The global match distance setting
        'effective_match_distance': effective_match_distance,  # The distance actually used (may be device's max_distance)
        'estimated_progress': estimated_progress  # Progress estimate for progress bar
    })

@app.route('/api/get_device_plots', methods=['GET'])
def api_get_device_plots():
    """Get plots for all events matching a specific device"""
    global analyzer, current_match_distance, plot_progress_scaling_factor, mode
    
    if analyzer is None or analyzer.events is None or len(analyzer.events) == 0:
        return jsonify({'success': False, 'message': 'No analysis or data available'})
    
    device_idx = request.args.get('device_idx', type=int)
    if device_idx is None:
        return jsonify({'success': False, 'message': 'Device ID required'})
    
    if device_idx < 0 or device_idx >= len(analyzer.device_labels):
        return jsonify({'success': False, 'message': f'Device {device_idx} not found'})
    
    # Set mode to browse when viewing a device (prevents auto-updates)
    mode = 'browse'
    
    # Get optional match_distance override, otherwise use device's max_distance
    match_distance_override = request.args.get('match_distance', type=float)
    
    try:
        start_time = time.time()
        
        # Get device profile and max distance
        device_profile = analyzer.device_profiles[device_idx]
        device_max_distance = analyzer.max_device_distances[device_idx]
        
        # Use override if provided, otherwise use device's max_distance
        effective_distance = match_distance_override if match_distance_override is not None else device_max_distance
        
        # Find all events matching this device
        from scipy.spatial.distance import cdist
        from poweranalyzer import myscaler
        
        device_profile_scaled = myscaler(device_profile.reshape(1, -1))
        all_events_scaled = myscaler(analyzer.events)
        distances = cdist(device_profile_scaled, all_events_scaled, metric='euclidean')[0]
        matching_mask = distances < effective_distance
        
        # Ensure mask length matches all arrays
        expected_length = len(analyzer.events)
        if len(matching_mask) != expected_length:
            logger.warning(f"Mask length mismatch: {len(matching_mask)} != {expected_length}, truncating mask")
            matching_mask = matching_mask[:expected_length] if len(matching_mask) > expected_length else np.pad(matching_mask, (0, expected_length - len(matching_mask)), constant_values=False)
        
        # Also ensure timestamps and other arrays match
        if len(analyzer.timestamps) != expected_length:
            logger.warning(f"Timestamps length mismatch: {len(analyzer.timestamps)} != {expected_length}")
            # Recreate analyzer to fix the mismatch
            return jsonify({'success': False, 'message': 'Data arrays are out of sync. Please refresh the page.'})
        
        matching_count = np.sum(matching_mask)
        
        # Calculate estimated progress based on matches found
        # Progress = (matching_count / total_events) * scaling_factor, capped at reasonable max
        estimated_progress = min(95, (matching_count * plot_progress_scaling_factor))
        
        if matching_count == 0:
            return jsonify({'success': False, 'message': f'No events found matching device {device_idx}'})
        
        # Calculate median pattern from matching events
        matching_events = analyzer.events[matching_mask]
        median_pattern = np.median(matching_events, axis=0)
        
        # Generate plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        device_label = analyzer.device_labels[device_idx] if device_idx < len(analyzer.device_labels) else f"Device {device_idx}"
        
        # Timeline plot
        timeline_title = f"Device {device_idx}: {device_label} ({matching_count} events)"
        fig_timeline, _ = analyzer.plot_timeline(matching_mask, title_label=timeline_title)
        timeline_img = io.BytesIO()
        fig_timeline.savefig(timeline_img, format='png', bbox_inches='tight', dpi=100)
        timeline_img.seek(0)
        timeline_base64 = base64.b64encode(timeline_img.getvalue()).decode('utf-8')
        plt.close(fig_timeline)
        
        # Power plot
        power_title = f"Device {device_idx}: {device_label} - {matching_count} matches"
        fig_power, _ = analyzer.plot_power(matching_mask, title_label=power_title,
                                                   device_pattern=median_pattern,
                                                   highlight_event_index=None)
        power_img = io.BytesIO()
        fig_power.savefig(power_img, format='png', bbox_inches='tight', dpi=100)
        power_img.seek(0)
        power_base64 = base64.b64encode(power_img.getvalue()).decode('utf-8')
        plt.close(fig_power)
        
        # Histogram plot
        hist_title = f"Device {device_idx}: {device_label} ({matching_count} events)"
        fig_hist, ax_hist = analyzer.plot_day_of_week_histogram(matching_mask, 
                                                                 title_label=hist_title)
        hist_img = io.BytesIO()
        fig_hist.savefig(hist_img, format='png', bbox_inches='tight', dpi=100)
        hist_img.seek(0)
        hist_base64 = base64.b64encode(hist_img.getvalue()).decode('utf-8')
        plt.close(fig_hist)
        
        # Calculate actual generation time and calibrate scaling factor
        actual_time = time.time() - start_time
        if matching_count > 0:
            time_per_matching_count = actual_time / matching_count
            plot_progress_scaling_factor = 0.9*plot_progress_scaling_factor + 0.1*time_per_matching_count
            logger.info(f"Plot progress scaling {plot_progress_scaling_factor:.3f} (time per matching count: {time_per_matching_count:.3f}s)")
        
        return jsonify({
            'success': True,
            'timeline': timeline_base64,
            'power': power_base64,
            'histogram': hist_base64,
            'device_idx': device_idx,
            'device_label': device_label,
            'max_distance': float(device_max_distance),
            'effective_distance': float(effective_distance),
            'off_delay': int(analyzer.off_delays[device_idx]) if device_idx < len(analyzer.off_delays) else -1,
            'matching_count': int(matching_count),
            'total_events': len(analyzer.events),
            'estimated_progress': estimated_progress  # Progress estimate for progress bar
        })
    except Exception as e:
        logger.error(f"Error generating device plots: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_device', methods=['POST'])
def api_update_device():
    """Update device properties (label, max_distance, off_delay)"""
    global analyzer, current_event_index, current_match_distance, mode
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis loaded'})
    
    data = request.json
    device_idx = data.get('device_idx')
    
    if device_idx is None:
        return jsonify({'success': False, 'message': 'Device ID required'})
    
    if device_idx < 0 or device_idx >= len(analyzer.device_labels):
        return jsonify({'success': False, 'message': f'Device {device_idx} not found'})
    
    try:
        # Update label if provided
        if 'label' in data:
            new_label = data['label'].strip()
            if new_label:
                analyzer.modify_device(device_idx, device_label=new_label)
        
        # Update max_distance if provided
        if 'max_distance' in data:
            new_max_distance = float(data['max_distance'])
            if new_max_distance > 0:
                analyzer.modify_device(device_idx, max_device_distance=new_max_distance)
        
        # Update off_delay if provided
        if 'off_delay' in data:
            new_off_delay = int(data['off_delay'])
            # Update in database
            analyzer.modify_device(device_idx, off_delay=new_off_delay)
        
        return jsonify({
            'success': True,
            'message': f'Device {device_idx} updated successfully'
        })
    except Exception as e:
        logger.error(f"Error updating device: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_device', methods=['POST'])
def api_delete_device():
    """Delete a device"""
    global analyzer, current_event_index, current_match_distance, mode
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis loaded'})
    
    data = request.json
    device_idx = data.get('device_idx')
    
    if device_idx is None:
        return jsonify({'success': False, 'message': 'Device ID required'})
    
    if device_idx < 0 or device_idx >= len(analyzer.device_labels):
        return jsonify({'success': False, 'message': f'Device {device_idx} not found'})
    
    try:
        device_label = analyzer.device_labels[device_idx] if device_idx < len(analyzer.device_labels) else f'Device {device_idx}'
        
        analyzer.delete_device(device_idx)
        
        return jsonify({
            'success': True,
            'message': f'Device "{device_label}" (ID: {device_idx}) deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting device: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/browse_event', methods=['POST'])
def api_browse_event():
    """Handle browse event actions (N, P, E, M, S)"""
    global analyzer, current_event_index, current_match_distance, mode
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No analysis loaded'})
    
    data = request.json
    action = data.get('action', '').upper()
    
    # Use event_index from request if provided, otherwise use global
    requested_index = data.get('event_index')
    if requested_index is not None:
        current_event_index = int(requested_index)
    
    try:
        # Handle navigation actions on server side
        if action == 'N':  # Next event
            mode = 'browse'  # Switch to browse mode when user navigates
            current_event_index += 1
            # Ensure index is valid after increment
            if current_event_index >= analyzer.n_events:
                current_event_index = analyzer.n_events - 1
        elif action == 'P':  # Previous event
            mode = 'browse'  # Switch to browse mode when user navigates
            current_event_index -= 1
            # Ensure index is valid after decrement
            if current_event_index < 0:
                current_event_index = 0
        elif action == 'L':  # Latest mode activated
            mode = 'monitor'  # Switch to monitor mode when latest is active
        
        # Ensure index is valid (for all actions)
        if current_event_index >= analyzer.n_events:
            current_event_index = analyzer.n_events - 1
        if current_event_index < 0:
            current_event_index = 0
        
        # Handle specific actions that need additional processing and return early
        if action == 'M':  # Change match distance (does not change mode)
            new_distance = data.get('distance')
            if new_distance is not None:
                current_match_distance = float(new_distance)
                if current_match_distance > 0:
                    logger.info(f"Match distance updated to {current_match_distance}")
                    return jsonify({'success': True, 'message': f'Match distance updated to {current_match_distance}'})
                else:
                    return jsonify({'success': False, 'message': 'Match distance must be greater than 0'})
            else:
                return jsonify({'success': False, 'message': 'Distance parameter required'})
        elif action == 'E' or action == 'S':  # Edit event (label cluster as device)
            current_cluster_id = analyzer.clusters[current_event_index]
            if current_cluster_id == -1:
                return jsonify({'success': False, 'message': 'Event has no cluster assigned'})
            
            label = data.get('label')
            if not label:
                return jsonify({'success': False, 'message': 'Label required'})
            
            device_id = analyzer._convert_cluster_to_device(current_cluster_id, label)
            return jsonify({
                'success': True,
                'message': f'Cluster {current_cluster_id} labeled as "{label}" (now device {device_id})',
                'device_id': device_id
            })
        elif action not in ('N', 'P', 'L'):  # Unknown action (but allow N, P, and L to continue)
            return jsonify({'success': False, 'message': f'Unknown action: {action}'})
        
        # For N, P, and L actions, return success (frontend will call loadPlots separately)
        return jsonify({'success': True, 'message': 'Action completed', 'event_index': current_event_index, 'mode': mode})
    except Exception as e:
        logger.error(f"Error in browse event action: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Initialize database
    current_event_index = None
    current_match_distance = 500
    mode = 'monitor'
    analyzer = PowerEventAnalyzer()
    analyzer.load_events()
    status = analyzer.get_status()
    logger.info(f"✅ Auto-loaded previous analysis (from {status['latest_analysis'].strftime('%Y-%m-%d %H:%M:%S') if status['latest_analysis'] else 'unknown'})")
    logger.info(f"   Found {status['number_of_events']} events")
    logger.info(f"   First event: {status['first_event'].strftime('%Y-%m-%d %H:%M:%S') if status['first_event'] else 'unknown'}")
    logger.info(f"   Last event: {status['last_event'].strftime('%Y-%m-%d %H:%M:%S') if status['last_event'] else 'unknown'}")
    current_event_index = analyzer.n_events - 1 if analyzer.n_events is not None else None
    if status['latest_analysis'] is not None:
        analyzer.load_results_from_database()
        current_match_distance = float(np.min(analyzer.max_device_distances))
    print("🚀 Starting Nonsense Power Analyzer Web Application...")
    print("=" * 60)
    print("Access the application at: http://localhost:8888")
    print("=" * 60)
    try:
        app.run(host='0.0.0.0', port=8888, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        if analyzer is not None:
            analyzer.close()

