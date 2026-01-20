import paho.mqtt.client as mqtt
import signal
import json
import logging

# MQTT settings
import os
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Detect paho-mqtt version for compatibility
# This code works with both paho-mqtt 1.6 and 2.1+
# When paho 2.0+ is detected, we use VERSION1 API which maintains backward compatibility
try:
    # paho-mqtt 2.0+ has CallbackAPIVersion
    _PAHO_VERSION_2 = hasattr(mqtt, 'CallbackAPIVersion')
except:
    _PAHO_VERSION_2 = False

load_dotenv()

# MQTT configuration - required if MQTT is used
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT_STR = os.getenv("MQTT_PORT")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# Validate MQTT_HOST (required for connection)
if MQTT_HOST is None:
    raise ValueError("MQTT_HOST environment variable is required but not set. Please set it in docker-compose.yml or a .env file, or disable MQTT functionality.")

# MQTT_PORT is optional, default to 1883 if not set
if MQTT_PORT_STR is None:
    MQTT_PORT = 1883
else:
    try:
        MQTT_PORT = int(MQTT_PORT_STR)
    except ValueError as e:
        raise ValueError(f"MQTT_PORT must be a valid integer. Got: '{MQTT_PORT_STR}'. Error: {e}")

# MQTT_USER and MQTT_PASSWORD are optional (can be None for unauthenticated MQTT)
# But we'll validate they're set together if one is provided
if (MQTT_USER is None) != (MQTT_PASSWORD is None):
    raise ValueError("Both MQTT_USER and MQTT_PASSWORD must be set together, or both left unset for unauthenticated MQTT.")


class HassMQTT:
    def __init__(self, mytopics):
        self.topics = mytopics
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        
        # Create client compatible with both paho 1.6 and 2.1
        # upgrade to mqtt.CallbackAPIVersion.VERSION2 when we can
        if _PAHO_VERSION_2:
            self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        else:
            self.mqttc = mqtt.Client()
        
        self.mqttc.username_pw_set(MQTT_USER, password=MQTT_PASSWORD)
        self.mqttc.on_connect = self._on_connect_wrapper
        self.mqttc.on_disconnect = self._on_disconnect_wrapper
        self.mqttc.on_message = self.on_message
        self.mqttc.connect(MQTT_HOST, port=MQTT_PORT, keepalive=60, bind_address="")
        logger.info('MQTT Config completed. Starting mqtt')
        self.mqttc.loop_start()
        for topic in self.topics:
            self.mqttc.will_set(self.topics[topic]["config_topic"], payload=None, qos=0, retain=False)
            config = json.dumps({k: self.topics[topic][k] for k in self.topics[topic].keys() if k != "config_topic"})
            self.send(self.topics[topic]["config_topic"], config, True)
        logger.info('HassMQTT config completed')

    def _on_connect_wrapper(self, client, userdata, flags, rc, *args, **kwargs):
        """Wrapper to handle both paho 1.6 and 2.1 callback signatures"""
        # VERSION1 API uses same signature as 1.6, so just pass through
        self.on_connect(client, userdata, flags, rc)

    def _on_disconnect_wrapper(self, client, userdata, rc, *args, **kwargs):
        """Wrapper to handle both paho 1.6 and 2.1 callback signatures"""
        # VERSION1 API uses same signature as 1.6, so just pass through
        self.on_disconnect(client, userdata, rc)

    def on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection - compatible with both paho 1.6 and 2.1"""
        logger.debug(f"MQTT connection returned result: {rc}")
        logger.info("MQTT connected")

    def on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection - compatible with both paho 1.6 and 2.1"""
        if rc != 0:
            logger.error(f"Unexpected disconnection from MQTT with reasoncode: {rc}.")


    def shutdown(self, _signum=None, _frame=None, _force=False):
        # uses signal to shut down and hard kill opened processes and self
        # _signum and _frame are provided when called as a signal handler but not used
        logger.info('Exiting....')
        for topic in self.topics:
            self.send(self.topics[topic]["config_topic"], "", False)
        # Handle loop_stop() differences between paho versions
        try:
            # Try paho 2.0+ signature (no parameters or with force)
            self.mqttc.loop_stop()
        except TypeError:
            # Fall back to paho 1.6 signature if needed
            self.mqttc.loop_stop(force=False)
        self.mqttc.disconnect()
        exit(0)
    

    def send(self, topic, payload, retain=True):
        try:
            logger.debug(f"Send to {topic} payload: {payload}")
            mqttmessageinfo = self.mqttc.publish(topic, payload=payload, qos=0, retain=retain)
            mqttmessageinfo.wait_for_publish()
        except Exception as ex:
            logger.error(f"MQTT Publish Failed: {ex}")
    
    def update(self, item, payload, retain=True):
        if item in self.topics:
            try:
                logger.debug(f"Update {item} to {payload}")
                mqttmessageinfo = self.mqttc.publish(self.topics[item]["state_topic"], payload=payload, qos=0, retain=retain)
                mqttmessageinfo.wait_for_publish()
            except Exception as ex:
                logger.error(f"MQTT Publish Failed: {ex}")
        else:
            logger.error(f"Item {item} not found in topics to update")


    def add_and_config_topic(self, topic_name, topic_dict):
        """
        Add a topic to self.topics dictionary and send its configuration.
        
        Args:
            topic_name: Name/key for the topic
            topic_dict: Dictionary containing topic configuration including config_topic
        """
        # Add topic to self.topics
        self.topics[topic_name] = topic_dict
        
        # Send configuration
        config = json.dumps({k: topic_dict[k] for k in topic_dict.keys() if k != "config_topic"})
        self.send(topic_dict["config_topic"], config, True)
    
    def reconfig(self, new_topics):
        """
        Append new topics to self.topics and send configuration for those new topics.
        
        Args:
            new_topics: Dictionary of topics to add (same format as __init__ parameter)
        """
        # Find topics that are new (not already in self.topics)
        new_topic_keys = [key for key in new_topics.keys() if key not in self.topics]
        
        if not new_topic_keys:
            logger.debug("No new topics to add in reconfig")
            return
        
        # Append new topics to self.topics
        for topic_name in new_topic_keys:
            self.topics[topic_name] = new_topics[topic_name]
            # Set will_set for the new topic
            self.mqttc.will_set(self.topics[topic_name]["config_topic"], payload=None, qos=0, retain=False)
            # Send configuration
            config = json.dumps({k: self.topics[topic_name][k] for k in self.topics[topic_name].keys() if k != "config_topic"})
            self.send(self.topics[topic_name]["config_topic"], config, True)
        
        logger.info(f'Reconfigured {len(new_topic_keys)} new topics')

    def on_message(self, _client, _userdata, _message):
        logger.info("receiving messages not implemented")


def main():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        #level=logging.DEBUG,  # capture everything from DEBUG upwards
        #format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        format ="%(message)s"
    )
    import time
    print('demo of how this can work')
    topics = {}
    print('recommend no spaces and all lower case for your MQTT topics')
    mqtt_topic_prefix = 'test'
    device ='test_device'
    entity_postfix = '_measurement'
    topics[device] = {
                    "config_topic": f"homeassistant/binary_sensor/{mqtt_topic_prefix}/{device}{entity_postfix}/config",
                    "state_topic": f"homeassistant/binary_sensor/{mqtt_topic_prefix}/{device}{entity_postfix}/state",
                    "icon": "mdi:flash",
                    "name": f'{device}{entity_postfix}',
                    "unique_id": f'{device}{entity_postfix}',
                    "device_class": "power"
                }
    hassmqtt = HassMQTT(topics)
    for message in range(5):
        if (message%2)==0:
            hassmqtt.send(topics[device]["state_topic"],"ON", False)   
        else:
            hassmqtt.send(topics[device]["state_topic"],"OFF", False)   
        time.sleep(10)
    hassmqtt.shutdown()            


if __name__ == "__main__":
    main()