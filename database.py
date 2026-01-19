#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save Power Event Data to SQLite Database

This program loads power event data using load_combined_data() from 
nonsense_power_analyzer.py and saves it to an SQLite database.
"""

import sqlite3
import os
import time
from datetime import datetime, timezone
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file name
DB_FILE = './data/power_events.db'

# EVENT_SIZE from powerAnalyzerv4.py
EVENT_SIZE = 20


class PowerEventDatabase:
    """
    Class for managing power event data in SQLite database.
    
    On initialization, connects to the database and creates/verifies the schema.
    Provides methods for adding events and managing the database.
    """
    
    def __init__(self, db_file=DB_FILE):
        """
        Initialize the database connection and create/verify schema.
        
        Args:
            db_file: Path to SQLite database file (default: DB_FILE)
        """
        self.db_file = db_file
        self.conn = None
        self._connect()
        self._create_schema()
        self._verify_status()
    
    def _connect(self):
        """Connect to the database"""
        try:
            # Ensure directory exists
            db_dir = os.path.dirname(self.db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # Enable threading support for SQLite (required for Flask multi-threaded environment)
            # check_same_thread=False allows the connection to be used from different threads
            # SQLite handles thread safety internally with proper locking
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            # Enable WAL mode for better concurrency
            self.conn.execute('PRAGMA journal_mode=WAL')
            logger.debug(f"Connected to database: {self.db_file}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _create_schema(self):
        """Create the events table if it doesn't exist"""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        cursor = self.conn.cursor()
        
        # Create table with timestamp and power columns (P01-P20)
        power_columns = ', '.join([f'P{i+1:02d} REAL' for i in range(EVENT_SIZE)])
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS events (
            timeStamp TIMESTAMP PRIMARY KEY,
            {power_columns}
        )
        """
        
        cursor.execute(create_table_sql)
        
        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timeStamp)
        """)
        
        # Create hourly_minimum_power table
        create_hourly_table_sql = """
        CREATE TABLE IF NOT EXISTS hourly_minimum_power (
            hour INTEGER PRIMARY KEY,
            minimum_power REAL NOT NULL
        )
        """
        cursor.execute(create_hourly_table_sql)
        
        # Create devices table for analysis results
        power_profile_columns = ', '.join([f'P{i+1} REAL' for i in range(EVENT_SIZE)])
        create_devices_table_sql = f"""
        CREATE TABLE IF NOT EXISTS devices (
            device_key INTEGER PRIMARY KEY,
            device_label TEXT NOT NULL,
            max_device_distance REAL NOT NULL,
            off_delay INTEGER NOT NULL,
            {power_profile_columns}
        )
        """
        cursor.execute(create_devices_table_sql)
        
        # Create index on device_key for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_device_key ON devices(device_key)
        """)
        
        # Create status table
        create_status_table_sql = """
        CREATE TABLE IF NOT EXISTS status (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
        cursor.execute(create_status_table_sql)
        
        # Create trigger to update status table when events are inserted
        # First, drop the trigger if it exists (to allow schema updates)
        cursor.execute("DROP TRIGGER IF EXISTS update_status_on_event_insert")
        
        # Create the trigger
        create_trigger_sql = """
        CREATE TRIGGER update_status_on_event_insert
        AFTER INSERT ON events
        BEGIN
            -- Update number_of_events with the total row count
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('number_of_events', CAST((SELECT COUNT(*) FROM events) AS TEXT));
            
            -- Update last_event only if new timestamp is greater than current last_event
            -- or if last_event doesn't exist
            INSERT OR IGNORE INTO status (key, value)
            VALUES ('last_event', NEW.timeStamp);
            
            UPDATE status
            SET value = NEW.timeStamp
            WHERE key = 'last_event' AND (
                value IS NULL OR 
                datetime(value) < datetime(NEW.timeStamp)
            );
            
            -- Update first_event only if new timestamp is less than current first_event
            -- or if first_event doesn't exist
            INSERT OR IGNORE INTO status (key, value)
            VALUES ('first_event', NEW.timeStamp);
            
            UPDATE status
            SET value = NEW.timeStamp
            WHERE key = 'first_event' AND (
                value IS NULL OR 
                datetime(value) > datetime(NEW.timeStamp)
            );
        END
        """
        cursor.execute(create_trigger_sql)
        
        # Create triggers to update status table when devices are modified
        # Drop existing triggers if they exist (to allow schema updates)
        cursor.execute("DROP TRIGGER IF EXISTS update_status_on_device_insert")
        cursor.execute("DROP TRIGGER IF EXISTS update_status_on_device_update")
        cursor.execute("DROP TRIGGER IF EXISTS update_status_on_device_delete")
        
        # Create INSERT trigger
        create_device_insert_trigger_sql = """
        CREATE TRIGGER update_status_on_device_insert
        AFTER INSERT ON devices
        BEGIN
            -- Update latest_analysis with current timestamp
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('latest_analysis', datetime('now'));
            
            -- Update n_devices with the count of rows in devices table
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('n_devices', CAST((SELECT COUNT(*) FROM devices) AS TEXT));
        END
        """
        cursor.execute(create_device_insert_trigger_sql)
        
        # Create UPDATE trigger
        create_device_update_trigger_sql = """
        CREATE TRIGGER update_status_on_device_update
        AFTER UPDATE ON devices
        BEGIN
            -- Update latest_analysis with current timestamp
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('latest_analysis', datetime('now'));
            
            -- Update n_devices with the count of rows in devices table
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('n_devices', CAST((SELECT COUNT(*) FROM devices) AS TEXT));
        END
        """
        cursor.execute(create_device_update_trigger_sql)
        
        # Create DELETE trigger
        create_device_delete_trigger_sql = """
        CREATE TRIGGER update_status_on_device_delete
        AFTER DELETE ON devices
        BEGIN
            -- Update latest_analysis with current timestamp
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('latest_analysis', datetime('now'));
            
            -- Update n_devices with the count of rows in devices table
            INSERT OR REPLACE INTO status (key, value)
            VALUES ('n_devices', CAST((SELECT COUNT(*) FROM devices) AS TEXT));
        END
        """
        cursor.execute(create_device_delete_trigger_sql)
        
        self.conn.commit()
        logger.info(f"Database schema created/verified: {self.db_file}")
    
    def add_event_row(self, timestamp, power_array):
        """
        Add a single row to the events table in the database.
        
        Args:
            timestamp: Timestamp Object of the event
            power_array: numpy array of length EVENT_SIZE (20) containing power values
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            ValueError: If power_array length doesn't match EVENT_SIZE
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        # Validate power_array length
        if len(power_array) != EVENT_SIZE:
            raise ValueError(f"power_array must have length {EVENT_SIZE}, got {len(power_array)}")
        
        # validate timestamp
        if not isinstance(timestamp, datetime) and hasattr(timestamp, 'isoformat'):
            raise ValueError(f"timestamp must be a datetime object, got {type(timestamp)}")
        
        # Ensure timestamp is timezone-aware (use local timezone if naive)
        if timestamp.tzinfo is None:
            # Get local timezone
            local_tz = datetime.now().astimezone().tzinfo
            timestamp = timestamp.replace(tzinfo=local_tz)
        
        timestamp_str = timestamp.isoformat()
        
        # Convert power_array to list of floats, handling NaN/None
        power_values = []
        for val in power_array:
            if np.isnan(val) if isinstance(val, (float, np.floating)) else (val is None):
                power_values.append(None)
            else:
                power_values.append(float(val))
        
        try:
            # Prepare insert statement
            power_cols = [f'P{i+1:02d}' for i in range(EVENT_SIZE)]
            placeholders = ', '.join(['?' for _ in range(EVENT_SIZE + 1)])  # +1 for timestamp
            insert_sql = f"""
                INSERT INTO events (timeStamp, {', '.join(power_cols)})
                VALUES ({placeholders})
            """
            
            # Prepare data tuple
            data_tuple = (timestamp_str,) + tuple(power_values)
            
            # Insert row
            cursor = self.conn.cursor()
            cursor.execute(insert_sql, data_tuple)
            self.conn.commit()
            
            logger.debug(f"Inserted event row with timestamp: {timestamp_str}")
            return True
            
        except sqlite3.IntegrityError as e:
            # Handle primary key constraint violation (duplicate timestamp)
            logger.warning(f"Event with timestamp {timestamp_str} already exists: {e}")
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Error inserting event row: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def load_data(self, existing_timestamps=None, existing_events=None):
        """
        Load data from the database.
        
        Args:
            existing_timestamps: Optional numpy array of existing timestamps. If provided,
                                only events with timestamps newer than the latest existing
                                timestamp will be returned.
            existing_events: Optional 2D numpy array of existing events (must match length
                           of existing_timestamps if provided).
        
        Returns:
            tuple: (timestamps, events) where:
                - timestamps: numpy array of timestamps (ordered)
                - events: 2D numpy array of shape (n_events, EVENT_SIZE) with power values
        
        Raises:
            RuntimeError: If database connection is not established
            ValueError: If existing_timestamps and existing_events have mismatched lengths
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        # Validate existing arrays if provided
        if existing_timestamps is not None and existing_events is not None:
            if len(existing_timestamps) != len(existing_events):
                raise ValueError(f"existing_timestamps and existing_events must have the same length, "
                               f"got {len(existing_timestamps)} and {len(existing_events)}")
        
        cursor = self.conn.cursor()
        
        # Build query - get all power columns
        power_cols = [f'P{i+1:02d}' for i in range(EVENT_SIZE)]
        select_cols = ['timeStamp'] + power_cols
        
        # Determine minimum timestamp if appending to existing data
        min_timestamp_str = None
        if existing_timestamps is not None and len(existing_timestamps) > 0:
            # Find the maximum timestamp in existing data
            max_existing = np.max(existing_timestamps)
            # Convert to ISO format string for SQL comparison
            min_timestamp_str = max_existing.isoformat()
        
        # Build SQL query
        if min_timestamp_str is not None:
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM events
                WHERE timeStamp > ?
                ORDER BY timeStamp ASC
            """
            cursor.execute(query, (min_timestamp_str,))
        else:
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM events
                ORDER BY timeStamp ASC
            """
            cursor.execute(query)
        
        # Fetch all results
        rows = cursor.fetchall()
        
        if len(rows) == 0:
            # No new data
            if existing_timestamps is not None and existing_events is not None:
                # Return existing arrays unchanged
                return existing_timestamps.copy(), existing_events.copy()
            else:
                # Return empty arrays
                return np.array([], dtype=float), np.empty((0, EVENT_SIZE), dtype=float)
        
        # Extract timestamps and power values
        new_timestamps = []
        new_events = []
        
        for row in rows:
            timestamp_str = row[0]
            power_values = row[1:]
            
            # Convert timestamp string to datetime object
            try:
                # Try parsing ISO format
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # Ensure timezone-aware (use local timezone if naive for backward compatibility)
                if timestamp.tzinfo is None:
                    # Get local timezone for backward compatibility with old data
                    local_tz = datetime.now().astimezone().tzinfo
                    timestamp = timestamp.replace(tzinfo=local_tz)
            except (ValueError, AttributeError) as e:
                logger.error(f"Could not parse timestamp: {timestamp_str}, error: {e}")
                continue  # Skip this row if timestamp can't be parsed
            
            new_timestamps.append(timestamp)
            
            # Convert power values to float array, handling None/NaN
            power_array = []
            for val in power_values:
                if val is None:
                    power_array.append(0.0)  # Use 0.0 for None values
                else:
                    try:
                        power_array.append(float(val))
                    except (ValueError, TypeError):
                        power_array.append(0.0)
            
            new_events.append(power_array)
        
        # Convert to numpy arrays
        new_timestamps_array = np.array(new_timestamps, dtype=object)
        new_events_array = np.array(new_events, dtype=float)
        
        # Ensure events array has correct shape
        if new_events_array.shape[1] != EVENT_SIZE:
            raise RuntimeError(f"Expected {EVENT_SIZE} power values per event, got {new_events_array.shape[1]}")
        
        # Combine with existing data if provided
        if existing_timestamps is not None and existing_events is not None:
            # Concatenate arrays
            combined_timestamps = np.concatenate([existing_timestamps, new_timestamps_array])
            combined_events = np.vstack([existing_events, new_events_array])
            
            # Sort by timestamp to ensure ordering
            sort_indices = np.argsort(combined_timestamps)
            combined_timestamps = combined_timestamps[sort_indices]
            combined_events = combined_events[sort_indices]
            
            logger.debug(f"Loaded {len(new_timestamps_array)} new events, "
                        f"total: {len(combined_timestamps)} events")
            return combined_timestamps, combined_events
        else:
            logger.debug(f"Loaded {len(new_timestamps_array)} events from database")
            return new_timestamps_array, new_events_array
    
    def update_hourly_minimum_power(self, hour, minimum_power):
        """
        Update or insert a row in the hourly_minimum_power table.
        
        Args:
            hour: Integer representing the hour (0-23)
            minimum_power: Minimum power value for that hour
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
            ValueError: If hour is not in valid range (0-23)
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        # Validate hour range
        if not isinstance(hour, int) or hour < 0 or hour > 23:
            raise ValueError(f"hour must be an integer between 0 and 23, got {hour}")
        
        try:
            # Use INSERT OR REPLACE to overwrite existing row
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO hourly_minimum_power (hour, minimum_power)
                VALUES (?, ?)
            """, (hour, float(minimum_power)))
            
            self.conn.commit()
            logger.debug(f"Updated hourly_minimum_power for hour {hour}: {minimum_power}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating hourly_minimum_power: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def get_baseline_power(self):
        """
        Get the minimum of the minimum_power column from hourly_minimum_power table.
        
        Returns:
            float: baseline_power or None if no data exists
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT min(minimum_power) FROM hourly_minimum_power
            """)
            
            result = cursor.fetchone()[0]
            
            # AVG returns None if no rows exist
            if result is None:
                logger.debug("No data in hourly_minimum_power table")
                return None
            
            return float(result)
            
        except Exception as e:
            logger.error(f"Error getting baseline_power: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_status(self):
        """
        Get status values from the status table.
        
        Returns:
            dict: Dictionary with keys 'latest_analysis', 'number_of_events', 'last_event', 'first_event', and 'n_devices'.
                  Values are datetime objects for timestamps, int for number_of_events/n_devices, or None if not found.
                  - latest_analysis: datetime object or None
                  - number_of_events: int or None
                  - last_event: datetime object or None
                  - first_event: datetime object or None
                  - n_devices: int or None
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        result = {
            'latest_analysis': None,
            'number_of_events': None,
            'last_event': None,
            'first_event': None,
            'n_devices': None
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Get all five status values
            cursor.execute("""
                SELECT key, value FROM status 
                WHERE key IN ('latest_analysis', 'number_of_events', 'last_event', 'first_event', 'n_devices')
            """)
            
            rows = cursor.fetchall()
            
            for key, value_str in rows:
                if key == 'latest_analysis' or key == 'last_event' or key == 'first_event':
                    # Parse timestamp string to datetime object
                    try:
                        # Try parsing ISO format (YYYY-MM-DD HH:MM:SS)
                        timestamp = datetime.fromisoformat(value_str.replace('Z', '+00:00'))
                        # Ensure timezone-aware (use local timezone if naive for backward compatibility)
                        if timestamp.tzinfo is None:
                            local_tz = datetime.now().astimezone().tzinfo
                            timestamp = timestamp.replace(tzinfo=local_tz)
                        result[key] = timestamp
                    except (ValueError, AttributeError) as e:
                        logger.error(f"Could not parse {key} timestamp: {value_str}, error: {e}")
                        result[key] = None
                elif key == 'number_of_events' or key == 'n_devices':
                    # Parse integer
                    try:
                        result[key] = int(value_str)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Could not parse {key}: {value_str}, error: {e}")
                        result[key] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    def _verify_status(self):
        """
        Verify and update status table based on actual events in the events table.
        Updates number_of_events, last_event, and first_event.
        Also ensures all timestamps in the events table have timezone information.
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        try:
            cursor = self.conn.cursor()
            
            # Get local timezone for adding to naive timestamps
            local_tz = datetime.now().astimezone().tzinfo
            
            # Check and update all timestamps in events table to ensure they have timezone
            cursor.execute("SELECT timeStamp FROM events")
            all_timestamps = cursor.fetchall()
            updated_count = 0
            
            for (timestamp_str,) in all_timestamps:
                try:
                    # Parse the timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # If timestamp is naive (no timezone), add local timezone and update
                    if timestamp.tzinfo is None:
                        timestamp_with_tz = timestamp.replace(tzinfo=local_tz)
                        timestamp_str_new = timestamp_with_tz.isoformat()
                        
                        # Update the timestamp in the database
                        cursor.execute("""
                            UPDATE events 
                            SET timeStamp = ? 
                            WHERE timeStamp = ?
                        """, (timestamp_str_new, timestamp_str))
                        updated_count += 1
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not parse timestamp {timestamp_str} for timezone verification: {e}")
                    continue
            
            if updated_count > 0:
                logger.info(f"Updated {updated_count} timestamps in events table to include timezone information")
                self.conn.commit()
            
            # Count total events
            cursor.execute("SELECT COUNT(*) FROM events")
            count_result = cursor.fetchone()
            number_of_events = count_result[0] if count_result else 0
            
            # Update number_of_events
            cursor.execute("""
                INSERT OR REPLACE INTO status (key, value)
                VALUES ('number_of_events', ?)
            """, (str(number_of_events),))
            
            # Get last_event (maximum timestamp)
            cursor.execute("SELECT MAX(timeStamp) FROM events")
            last_event_result = cursor.fetchone()
            last_event = last_event_result[0] if last_event_result and last_event_result[0] else None
            
            if last_event:
                cursor.execute("""
                    INSERT OR REPLACE INTO status (key, value)
                    VALUES ('last_event', ?)
                """, (str(last_event),))
            else:
                # If no events, remove last_event
                cursor.execute("DELETE FROM status WHERE key = 'last_event'")
            
            # Get first_event (minimum timestamp)
            cursor.execute("SELECT MIN(timeStamp) FROM events")
            first_event_result = cursor.fetchone()
            first_event = first_event_result[0] if first_event_result and first_event_result[0] else None
            
            if first_event:
                cursor.execute("""
                    INSERT OR REPLACE INTO status (key, value)
                    VALUES ('first_event', ?)
                """, (str(first_event),))
            else:
                # If no events, remove first_event
                cursor.execute("DELETE FROM status WHERE key = 'first_event'")
            
            # Count devices
            cursor.execute("SELECT COUNT(*) FROM devices")
            devices_result = cursor.fetchone()
            n_devices = devices_result[0] if devices_result else 0
            
            # Update n_devices
            cursor.execute("""
                INSERT OR REPLACE INTO status (key, value)
                VALUES ('n_devices', ?)
            """, (str(n_devices),))
            
            self.conn.commit()
            logger.debug(f"Verified status: {number_of_events} events, {n_devices} devices, first_event={first_event}, last_event={last_event}")
            
        except Exception as e:
            logger.error(f"Error verifying status: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            raise
    
    def save_analysis(self, analysis_data):
        """
        Save analysis results (devices) to the devices table.
        
        Args:
            analysis_data: Dictionary containing:
                - device_key: 1D numpy array of integers (device IDs)
                - device_label: 1D numpy array or list of strings (device labels)
                - max_device_distance: 1D numpy array of floats (max distances)
                - off_delay: 1D numpy array of integers (off delays)
                - profile: 2D numpy array of shape (n_devices, EVENT_SIZE) with power profiles (P1-P20)
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
            ValueError: If input data is invalid or arrays have mismatched lengths
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        # Extract data from dictionary
        device_key = analysis_data.get('device_key')
        device_label = analysis_data.get('device_label')
        max_device_distance = analysis_data.get('max_device_distance')
        off_delay = analysis_data.get('off_delay')
        profile = analysis_data.get('profile')
        
        # Validate all required fields are present
        if device_key is None or device_label is None or max_device_distance is None or \
           off_delay is None or profile is None:
            raise ValueError("All fields (device_key, device_label, max_device_distance, off_delay, profile) must be provided")
        
        # Convert to numpy arrays if needed
        device_key = np.asarray(device_key, dtype=int)
        max_device_distance = np.asarray(max_device_distance, dtype=float)
        off_delay = np.asarray(off_delay, dtype=int)
        profile = np.asarray(profile, dtype=float)
        
        # Convert device_label to list if it's a numpy array
        if isinstance(device_label, np.ndarray):
            device_label = device_label.tolist()
        elif not isinstance(device_label, (list, tuple)):
            raise ValueError("device_label must be a list, tuple, or numpy array")
        
        # Validate array lengths match
        n_devices = len(device_key)
        if len(device_label) != n_devices:
            raise ValueError(f"device_label length ({len(device_label)}) doesn't match device_key length ({n_devices})")
        if len(max_device_distance) != n_devices:
            raise ValueError(f"max_device_distance length ({len(max_device_distance)}) doesn't match device_key length ({n_devices})")
        if len(off_delay) != n_devices:
            raise ValueError(f"off_delay length ({len(off_delay)}) doesn't match device_key length ({n_devices})")
        if profile.shape[0] != n_devices:
            raise ValueError(f"profile first dimension ({profile.shape[0]}) doesn't match device_key length ({n_devices})")
        if profile.shape[1] != EVENT_SIZE:
            raise ValueError(f"profile second dimension ({profile.shape[1]}) must be {EVENT_SIZE}, got {profile.shape[1]}")
        
        try:
            cursor = self.conn.cursor()
            
            # Clear existing devices table
            cursor.execute("DELETE FROM devices")
            
            # Prepare insert statement
            power_cols = [f'P{i+1}' for i in range(EVENT_SIZE)]
            placeholders = ', '.join(['?' for _ in range(4 + EVENT_SIZE)])  # 4 for key, label, distance, delay + EVENT_SIZE for profile
            insert_sql = f"""
                INSERT INTO devices (device_key, device_label, max_device_distance, off_delay, {', '.join(power_cols)})
                VALUES ({placeholders})
            """
            
            # Insert each device
            for i in range(n_devices):
                # Prepare data tuple: key, label, distance, delay, then P1-P20
                data_tuple = (
                    int(device_key[i]),
                    str(device_label[i]),
                    float(max_device_distance[i]),
                    int(off_delay[i]),
                ) + tuple(float(profile[i, j]) for j in range(EVENT_SIZE))
                
                cursor.execute(insert_sql, data_tuple)
            
            self.conn.commit()
            logger.info(f"Saved {n_devices} devices to analysis table")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis data: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def load_analysis(self):
        """
        Load analysis results (devices) from the devices table.
        
        Returns:
            dict: Dictionary containing:
                - device_key: 1D numpy array of integers (device IDs)
                - device_label: 1D numpy array of strings (device labels)
                - max_device_distance: 1D numpy array of floats (max distances)
                - off_delay: 1D numpy array of integers (off delays)
                - profile: 2D numpy array of shape (n_devices, EVENT_SIZE) with power profiles (P1-P20)
            Returns None if no devices are found
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        try:
            cursor = self.conn.cursor()
            
            # Build query to get all device columns
            power_cols = [f'P{i+1}' for i in range(EVENT_SIZE)]
            select_cols = ['device_key', 'device_label', 'max_device_distance', 'off_delay'] + power_cols
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM devices
                ORDER BY device_key ASC
            """
            cursor.execute(query)
            
            rows = cursor.fetchall()
            
            if len(rows) == 0:
                logger.debug("No devices found in analysis table")
                return None
            
            # Extract data from rows
            device_keys = []
            device_labels = []
            max_device_distances = []
            off_delays = []
            profiles = []
            
            for row in rows:
                device_keys.append(int(row[0]))
                device_labels.append(str(row[1]))
                max_device_distances.append(float(row[2]))
                off_delays.append(int(row[3]))
                # Extract P1-P20 values
                profile_values = [float(row[4 + j]) for j in range(EVENT_SIZE)]
                profiles.append(profile_values)
            
            # Convert to numpy arrays
            result = {
                'device_key': np.array(device_keys, dtype=int),
                'device_label': np.array(device_labels, dtype=object),
                'max_device_distance': np.array(max_device_distances, dtype=float),
                'off_delay': np.array(off_delays, dtype=int),
                'profile': np.array(profiles, dtype=float)
            }
            
            logger.debug(f"Loaded {len(device_keys)} devices from analysis table")
            return result
            
        except Exception as e:
            logger.error(f"Error loading analysis data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_device(self, device_key, device_label, max_device_distance, off_delay, profile):
        """
        Add a single device to the devices table.
        
        Args:
            device_key: Integer device ID
            device_label: String device label
            max_device_distance: Float max device distance
            off_delay: Integer off delay
            profile: 1D numpy array of length EVENT_SIZE with power profile (P1-P20)
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
            ValueError: If profile length doesn't match EVENT_SIZE or device_key already exists
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        # Convert profile to numpy array and validate
        profile = np.asarray(profile, dtype=float)
        if len(profile) != EVENT_SIZE:
            raise ValueError(f"profile must have length {EVENT_SIZE}, got {len(profile)}")
        
        # Validate other inputs
        device_key = int(device_key)
        device_label = str(device_label)
        max_device_distance = float(max_device_distance)
        off_delay = int(off_delay)
        
        try:
            cursor = self.conn.cursor()
            
            # Prepare insert statement
            power_cols = [f'P{i+1}' for i in range(EVENT_SIZE)]
            placeholders = ', '.join(['?' for _ in range(4 + EVENT_SIZE)])
            insert_sql = f"""
                INSERT INTO devices (device_key, device_label, max_device_distance, off_delay, {', '.join(power_cols)})
                VALUES ({placeholders})
            """
            
            # Prepare data tuple: key, label, distance, delay, then P1-P20
            data_tuple = (
                device_key,
                device_label,
                max_device_distance,
                off_delay,
            ) + tuple(float(profile[j]) for j in range(EVENT_SIZE))
            
            cursor.execute(insert_sql, data_tuple)
            self.conn.commit()
            
            logger.info(f"Added device {device_key} ({device_label}) to analysis table")
            return True
            
        except sqlite3.IntegrityError as e:
            # Handle primary key constraint violation (duplicate device_key)
            logger.warning(f"Device with key {device_key} already exists: {e}")
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Error adding device: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def modify_device(self, device_key, device_label=None, max_device_distance=None, off_delay=None):
        """
        Modify an existing device in the devices table.
        
        Args:
            device_key: Integer device ID (required, identifies which device to modify)
            device_label: Optional string device label to update
            max_device_distance: Optional float max device distance to update
            off_delay: Optional integer off delay to update
            
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
            ValueError: if device_key doesn't exist
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        device_key = int(device_key)
        
        # Check if at least one field is being updated
        if device_label is None and max_device_distance is None and off_delay is None:
            logger.warning("No fields provided to update")
            return False
        
        try:
            cursor = self.conn.cursor()
            
            # Build UPDATE statement dynamically based on provided fields
            update_parts = []
            data_tuple = []
            
            if device_label is not None:
                update_parts.append("device_label = ?")
                data_tuple.append(str(device_label))
            
            if max_device_distance is not None:
                update_parts.append("max_device_distance = ?")
                data_tuple.append(float(max_device_distance))
            
            if off_delay is not None:
                update_parts.append("off_delay = ?")
                data_tuple.append(int(off_delay))
                        # Add device_key to WHERE clause
            data_tuple.append(device_key)
            
            update_sql = f"""
                UPDATE devices
                SET {', '.join(update_parts)}
                WHERE device_key = ?
            """
            
            cursor.execute(update_sql, tuple(data_tuple))
            
            if cursor.rowcount == 0:
                logger.warning(f"Device with key {device_key} not found")
                self.conn.rollback()
                return False
            
            self.conn.commit()
            logger.info(f"Modified device {device_key} in analysis table")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying device: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def delete_device(self, device_key):
        """
        Delete a device from the devices table.
        
        Args:
            device_key: Integer device ID to delete
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        device_key = int(device_key)
        
        try:
            cursor = self.conn.cursor()
            
            delete_sql = """
                DELETE FROM devices
                WHERE device_key = ?
            """
            
            cursor.execute(delete_sql, (device_key,))
            
            if cursor.rowcount == 0:
                logger.warning(f"Device with key {device_key} not found")
                self.conn.rollback()
                return False
            
            self.conn.commit()
            logger.info(f"Deleted device {device_key} from analysis table")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting device: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def export_devices(self, filename):
        """
        Export devices table to a CSV file.
        
        Args:
            filename: Path to the output CSV file
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        try:
            import csv
            
            cursor = self.conn.cursor()
            
            # Get all devices
            power_cols = [f'P{i+1}' for i in range(EVENT_SIZE)]
            select_cols = ['device_key', 'device_label', 'max_device_distance', 'off_delay'] + power_cols
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM devices
                ORDER BY device_key ASC
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if len(rows) == 0:
                logger.warning("No devices to export")
                return False
            
            # Write to CSV file
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(select_cols)
                
                # Write data rows
                for row in rows:
                    writer.writerow(row)
            
            logger.info(f"Exported {len(rows)} devices to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting devices: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def import_devices(self, filename):
        """
        Import devices from a CSV file into the devices table.
        Replaces all existing devices.
        
        Args:
            filename: Path to the input CSV file
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        
        try:
            import csv
            
            cursor = self.conn.cursor()
            
            # Read CSV file
            devices_data = []
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                
                # Read header
                header = next(reader)
                expected_cols = ['device_key', 'device_label', 'max_device_distance', 'off_delay'] + [f'P{i+1}' for i in range(EVENT_SIZE)]
                
                if header != expected_cols:
                    raise ValueError(f"CSV header mismatch. Expected {expected_cols}, got {header}")
                
                # Read data rows
                for row in reader:
                    if len(row) != len(expected_cols):
                        logger.warning(f"Skipping row with incorrect number of columns: {row}")
                        continue
                    
                    devices_data.append(row)
            
            if len(devices_data) == 0:
                logger.warning("No devices to import")
                return False
            
            # Clear existing devices
            cursor.execute("DELETE FROM devices")
            
            # Prepare insert statement
            power_cols = [f'P{i+1}' for i in range(EVENT_SIZE)]
            placeholders = ', '.join(['?' for _ in range(4 + EVENT_SIZE)])
            insert_sql = f"""
                INSERT INTO devices (device_key, device_label, max_device_distance, off_delay, {', '.join(power_cols)})
                VALUES ({placeholders})
            """
            
            # Insert devices
            for row in devices_data:
                device_key = int(row[0])
                device_label = str(row[1])
                max_device_distance = float(row[2])
                off_delay = int(row[3])
                power_values = [float(row[4 + i]) for i in range(EVENT_SIZE)]
                
                data_tuple = (device_key, device_label, max_device_distance, off_delay) + tuple(power_values)
                cursor.execute(insert_sql, data_tuple)
            
            self.conn.commit()
            logger.info(f"Imported {len(devices_data)} devices from {filename}")
            return True
            
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return False
        except Exception as e:
            logger.error(f"Error importing devices: {e}")
            import traceback
            traceback.print_exc()
            self.conn.rollback()
            return False
    
    def close(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.debug("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

