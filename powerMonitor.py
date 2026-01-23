# -*- coding: utf-8 -*-
"""
Spyder Editor

This pulls data from enphase once per second
looks for events
stores the events in an SQLite database using add_event_row()

"""


import requests
import time
from datetime import datetime
import numpy as np
import os
import logging
from dotenv import load_dotenv
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # type: ignore
import poweranalyzer as pa
from HassMQTT import HassMQTT
from database import PowerEventDatabase

# Set up logger
logger = logging.getLogger(__name__)

load_dotenv()

# Required environment variables with validation
ENPHASE_EMAIL = os.getenv("ENPHASE_EMAIL")
if ENPHASE_EMAIL is None:
    raise ValueError("ENPHASE_EMAIL environment variable is required but not set. Please set it in docker-compose.yml or a .env file.")

ENPHASE_PASSWORD = os.getenv("ENPHASE_PASSWORD")
if ENPHASE_PASSWORD is None:
    raise ValueError("ENPHASE_PASSWORD environment variable is required but not set. Please set it in docker-compose.yml or a .env file.")

ENVOY_IP = os.getenv("ENVOY_IP")
if ENVOY_IP is None:
    raise ValueError("ENVOY_IP environment variable is required but not set. Please set it in docker-compose.yml or a .env file.")

ENVOY_SERIAL = os.getenv("ENVOY_SERIAL")
if ENVOY_SERIAL is None:
    raise ValueError("ENVOY_SERIAL environment variable is required but not set. Please set it in docker-compose.yml or a .env file.")

# GRID_EID is not provided, choose the first meter found
GRID_EID = os.getenv("GRID_EID")
# SOLAR_EID is only needed if you have solar behind the grid meter
SOLAR_EID = os.getenv("SOLAR_EID")

TRIGGER = 20 # 20W change will trigger an event, 5 looks like noise all day long, 10W also looks like too much noise

INTERVAL = 1 # we do 1 second interval sampling


class EnphaseClient:
    """
    Client for interacting with Enphase Envoy API.
    
    Handles authentication and data retrieval from Enphase Envoy device.
    """
    
    def __init__(self, email, password, envoy_ip, envoy_serial):
        """
        Initialize the Enphase client.
        
        Args:
            email: Enphase account email
            password: Enphase account password
            envoy_ip: Local IP address of the Envoy device
            envoy_serial: Serial number of the Envoy device
        """
        self.email = email
        self.password = password
        self.envoy_ip = envoy_ip
        self.envoy_serial = envoy_serial
        self.token = None
        self.url = f"http://{envoy_ip}/ivp/meters/readings"
        self.grid_eid = GRID_EID
        self.solar_eid = SOLAR_EID
    
    def get_token(self):
        """
        Get or refresh the authentication token.
        
        Returns:
            str: JWT access token
        
        Raises:
            requests.HTTPError: If authentication fails
        """
        # Get session ID
        r = requests.post(
            'https://enlighten.enphaseenergy.com/login/login.json?',
            data={'user[email]': self.email, 'user[password]': self.password}
        )
        r.raise_for_status()
        session_id = r.json().get('session_id')
        
        # Get token
        token_resp = requests.post(
            'https://entrez.enphaseenergy.com/tokens',
            json={'session_id': session_id, 'serial_num': self.envoy_serial, 'username': self.email}
        )
        token_resp.raise_for_status()
        self.token = token_resp.text  # This is your JWT access token
        return self.token
    
    def get_consumption_data(self):
        """
        Get current consumption data from the Envoy.
        
        Returns:
            tuple: (timestamp, total_power) where:
                - timestamp: Unix timestamp
                - total_power: Total power (grid + solar) in watts
            Returns None if an error occurs
        """
        # Ensure we have a token
        if self.token is None:
            try:
                self.get_token()
            except Exception as e:
                logger.error(f"Error getting token: {e}")
                return None
        
        try:
            r = requests.get(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                verify=False,
                timeout=5
            )
            r.raise_for_status()
            data = r.json()
            
            # Filter only consumption meters
            #  "eid": GRID_EID is consumption
            #  "eid": SOLAR_EID is solar
            if self.grid_eid is not None:
                consumption_meter = [m for m in data if m.get("eid") == int(self.grid_eid)]
            else:
                consumption_meter = None
            if self.solar_eid is not None:
                solar_meter = [m for m in data if m.get("eid") == int(self.solar_eid)]
            else:
                solar_meter = None
            
            if not consumption_meter:
                logger.error(f"Consumption meter (eid: {self.grid_eid}) not found in data")
                return None
            
            timestamp = consumption_meter[0].get('timestamp')
            grid_power = consumption_meter[0].get('activePower', 0)
            solar_power = solar_meter[0].get('activePower', 0) if solar_meter else 0
            
            # Create timezone-aware datetime from Unix timestamp
            dt = datetime.fromtimestamp(timestamp)
            # Make it timezone-aware using local timezone
            if dt.tzinfo is None:
                local_tz = datetime.now().astimezone().tzinfo
                dt = dt.replace(tzinfo=local_tz)
            return dt, grid_power + solar_power
        except requests.HTTPError as e:
            # Token might be expired, try to refresh
            if e.response.status_code == 401:
                logger.warning("Token expired, attempting to refresh...")
                try:
                    self.get_token()
                    # Retry the request
                    r = requests.get(
                        self.url,
                        headers={"Authorization": f"Bearer {self.token}"},
                        verify=False,
                        timeout=5
                    )
                    r.raise_for_status()
                    data = r.json()
                    if self.grid_eid is not None:
                        consumption_meter = [m for m in data if m.get("eid") == int(self.grid_eid)]
                    else:
                        consumption_meter = None
                    if self.solar_eid is not None:
                        solar_meter = [m for m in data if m.get("eid") == int(self.solar_eid)]
                    else:
                        solar_meter = None
                    if not consumption_meter:
                        logger.error("Consumption meter not found after token refresh")
                        return None
                    timestamp = consumption_meter[0].get('timestamp')
                    grid_power = consumption_meter[0].get('activePower', 0)
                    solar_power = solar_meter[0].get('activePower', 0) if solar_meter else 0
                    # Create timezone-aware datetime from Unix timestamp
                    dt = datetime.fromtimestamp(timestamp)
                    # Make it timezone-aware using local timezone
                    if dt.tzinfo is None:
                        local_tz = datetime.now().astimezone().tzinfo
                        dt = dt.replace(tzinfo=local_tz)
                    return dt, grid_power + solar_power

                except Exception as retry_error:
                    logger.error(f"Error after token refresh: {retry_error}")
                    return None
            else:
                logger.error(f"HTTP error fetching data: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching consumption data: {e}")
            return None
    
    def report_all_meters(self):
        """
        Get and report all meters found in the Envoy API response.
        
        Logs all meters with their details (eid, activePower, timestamp, etc.)
        
        Returns:
            list: List of all meter dictionaries found, or None if an error occurs
        """
        # Ensure we have a token
        if self.token is None:
            try:
                self.get_token()
            except Exception as e:
                logger.error(f"Error getting token: {e}")
                return None
        
        try:
            r = requests.get(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                verify=False,
                timeout=5
            )
            r.raise_for_status()
            data = r.json()
            
            # Log all meters found
            logger.info(f"Found {len(data)} meter(s) in Envoy response:")
            logger.info('Set GRID_EID and SOLAR_EID in the environment variables to use the correct meters')
            logger.info(f'only set SOLAR_EID if the solar is downstream of the consumption meter')
            for i, meter in enumerate(data):
                eid = meter.get("eid")
                if self.grid_eid == None:
                    self.grid_eid = eid
                    logger.info(f"Setting GRID_EID to first meter found: {self.grid_eid}")
                if self.grid_eid is not None and int(self.grid_eid) == eid:
                    comment = " used for consumption meter"
                elif self.solar_eid is not None and int(self.solar_eid) == eid:
                    comment = " used for solar meter (added to consumption meter)"
                else:
                    comment = " unused meter"
                active_power = meter.get("activePower", "N/A")
                timestamp = meter.get("timestamp", "N/A")
                logger.info(f"  Meter {i+1}: {comment} eid={eid}, activePower={active_power}W, timestamp={timestamp}")
            return data
        except requests.HTTPError as e:
            # Token might be expired, try to refresh
            if e.response.status_code == 401:
                logger.warning("Token expired, attempting to refresh...")
                try:
                    self.get_token()
                    # Retry the request
                    r = requests.get(
                        self.url,
                        headers={"Authorization": f"Bearer {self.token}"},
                        verify=False,
                        timeout=5
                    )
                    r.raise_for_status()
                    data = r.json()
                    
                    # Log all meters found
                    logger.info(f"Found {len(data)} meter(s) in Envoy response:")
                    for i, meter in enumerate(data):
                        eid = meter.get("eid")
                        active_power = meter.get("activePower", "N/A")
                        timestamp = meter.get("timestamp", "N/A")
                        logger.info(f"  Meter {i+1}: eid={eid}, activePower={active_power}W, timestamp={timestamp}")
                    return data
                except Exception as retry_error:
                    logger.error(f"Error after token refresh: {retry_error}")
                    return None
            else:
                logger.error(f"HTTP error fetching meter data: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching meter data: {e}")
            return None


class LoopTimer:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.next_time = time.time()
        self.average_sleep_time = interval  # Exponential moving average of sleep times

    def wait(self):
        self.next_time += self.interval
        sleep_time = self.next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            self.next_time = time.time()
            logger.warning(f"that took too long: {self.average_sleep_time=}")
        self.average_sleep_time = 0.99 * self.average_sleep_time + 0.01 * (sleep_time if sleep_time > 0 else 0.0)
    

class PowerTracking:
    """
    Tracks minimum power per hour and calculates average minimum power.
    Updates database and MQTT when hour rolls over.
    """
    
    def __init__(self, db, hassmqtt):
        """
        Initialize PowerTracking.
        
        Args:
            db: PowerEventDatabase instance
            hassmqtt: HassMQTT instance for sending updates
        """
        self.db = db
        self.hassmqtt = hassmqtt
        self.current_hour = datetime.now().hour
        self.hourly_minimum = float('inf')
        self.topic_names = ["baseline_power", "unknown_power"]
        self.baseline_power = 0
        self.power_sum = 0
        self.power_count = 1
        self.current_minutes = datetime.now().minute
        
        # Add average_minimum_power topic to topics dictionary in hassmqtt
        mqtt_topic_prefix = "powermonitor"
        for topic_name in self.topic_names:
            topic_dict = {
                "config_topic": f"homeassistant/sensor/{mqtt_topic_prefix}/{topic_name}/config",
                "state_topic": f"homeassistant/sensor/{mqtt_topic_prefix}/{topic_name}/state",
                "icon": "mdi:flash",
                "name": topic_name,
                "unique_id": topic_name,
                "device_class": "power",
                "unit_of_measurement": "W",
                "device": {
                    "name": "Power Monitor",
                    "model": "Power Monitor 1.0",
                    "manufacturer": "nils154",
                    "identifiers": 'Power Monitor',
                }
            }
            # Add and configure the new topic
            hassmqtt.add_and_config_topic(topic_name, topic_dict)
        
        # Get initial average
        self.baseline_power = self.db.get_baseline_power()
        if self.baseline_power is not None:
            hassmqtt.update('baseline_power', str(self.baseline_power), False)
    
    
    def update(self, timestamp, activePower, currently_on):
        """
        Update minimum power tracking for the current hour.
        
        Args:
            timestamp: Unix timestamp
            activePower: Current active power value
        """
        hour = timestamp.hour
        minutes = timestamp.minute
        self.hourly_minimum = min(self.hourly_minimum, activePower)

        # check if minutes has rolled over
        if minutes != self.current_minutes:
            unknown_power = self.power_sum / self.power_count
            self.power_sum = 0
            self.power_count = 0
            self.current_minutes = minutes
            for _device, (_scheduled_time, dev_avg_power) in currently_on.items():
                unknown_power -= dev_avg_power
            if self.baseline_power is not None:
                unknown_power = unknown_power-self.baseline_power 
            else:
                logger.error("Baseline power is None, cannot calculate unknown power")
                unknown_power = 0
            if unknown_power < 0:
                unknown_power = 0
            self.hassmqtt.update("unknown_power", str(unknown_power), False)
            logger.debug(f"Unknown power: {unknown_power}")
        self.power_sum += activePower
        self.power_count += 1
        # Check if hour has rolled over
        if hour != self.current_hour:
            # Hour rolled over, save previous hour's minimum
            self.db.update_hourly_minimum_power(self.current_hour, self.hourly_minimum)
            logger.debug(f"Saved minimum power for hour {self.current_hour}: {self.hourly_minimum}")
            
            # Get updated average
            self.baseline_power = self.db.get_baseline_power()
            if self.baseline_power is not None:
                self.hassmqtt.update("baseline_power", str(self.baseline_power ), False)
                logger.debug(f"Updated baseline_power {self.baseline_power }")
            
            # Reset for new hour
            self.hourly_minimum = activePower
            self.current_hour = hour
     
             
def main():
    # Configure logging - only show messages from powerMonitor and poweranalyzer
    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(
        level=logging.WARNING,  # Suppress INFO/DEBUG from third-party modules
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Set specific loggers to INFO level
    logging.getLogger('powerMonitor').setLevel(logging.DEBUG)
    logging.getLogger('poweranalyzer').setLevel(logging.DEBUG)
    logging.getLogger('__main__').setLevel(logging.DEBUG)
    
    # Initialize database connection
    db = PowerEventDatabase()
    # Initialize Enphase client
    enphase = EnphaseClient(ENPHASE_EMAIL, ENPHASE_PASSWORD, ENVOY_IP, ENVOY_SERIAL)
    enphase.report_all_meters()
    # load known labeled events
    analyzer = pa.PowerEventAnalyzer()
    topics = analyzer.load_results_from_database()
    hassmqtt = HassMQTT(topics)
    # Initialize power tracking
    power_tracking = PowerTracking(db, hassmqtt)
   
    # Dictionary to store currently on devices and their scheduled times and average power
    currently_on = {}
    # array to store power data for an event
    power_array = np.zeros(pa.EVENT_SIZE)
    # get initial consumption data
    result = enphase.get_consumption_data()
    if result is None:
        logger.error("Failed to fetch initial consumption data. Exiting.")
        exit(1)
    timeStamp, lastActivePower = result
    sample = 0
    # Continuous loop at ~1 Hz
    timer = LoopTimer(INTERVAL)
    logger.info('Starting loop')
    while True:
        timer.wait()
        result = enphase.get_consumption_data()
        if result is None:
            continue
        timeStamp,activePower = result 
        #logger.debug(f'Consumption data: {activePower}')
        if sample>0: # we are in a collection loop
            power_array[sample] = activePower-lastActivePower
            sample += 1
            dev_avg_power = 0.8*dev_avg_power + 0.2*(activePower-lastActivePower)
            if sample == pa.EVENT_SIZE:
                logger.debug('Event captured')
                sample = 0
                lastActivePower = activePower
                device_label, scheduled_time = analyzer.match(timeStamp, power_array, dev_avg_power, hassmqtt)
                if device_label is not None and scheduled_time is not None:
                    currently_on[device_label] = (scheduled_time, dev_avg_power)
                # Ensure event_timestamp is timezone-aware before saving
                if event_timestamp.tzinfo is None:
                    local_tz = datetime.now().astimezone().tzinfo
                    event_timestamp = event_timestamp.replace(tzinfo=local_tz)
                # Save event to database
                db.add_event_row(event_timestamp, power_array)
        elif abs(activePower-lastActivePower) > TRIGGER: # start a new collection loop
            logger.info(f'Event triggered')
            event_timestamp = timeStamp
            power_array[0] = activePower-lastActivePower
            dev_avg_power = activePower-lastActivePower
            sample = 1
        else:
            #not in a critical sample loop, update power tracking asynchronously
            power_tracking.update(timeStamp, activePower, currently_on)
            # Check for queued off actions that are due
            completed_actions = []
            for device, (scheduled_time, _) in currently_on.items():
                if timeStamp >= scheduled_time:
                    # Action is due, send OFF message
                    hassmqtt.update(device, "OFF", False)
                    hassmqtt.update(f'{device}_power', 0, False)
                    logger.info(f'Sent queued OFF action for {device}')
                    completed_actions.append(device)
            # Remove processed actions from queue
            for device in completed_actions:
                del currently_on[device]
            lastActivePower = activePower      


if __name__ == "__main__":
    main()