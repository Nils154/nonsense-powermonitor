Now that Sense no longer sells the orange powermonitor, and the green powermonitor from Scheiderhome doesn't work with homeassistant, I decided to try my own.
This app uses a once-per-second power measurement from my enphase setup. (Other hardware can easily be added in powerMonitor.py)
The power has to change by 20W to trigger and event. Depending on your setup, maybe this needs to be higher.
TVs and Computers use very unpredictable amounts of power and look like noise.
LED use very little power and blend in with the TVs and Computers. No hope in detecting those.
(It could be a future feature to add MQTT inputs from smartplugs through homeassistant)
After an event is triggered, 20 seconds is recorded and saved as an event.
The event is compared to 'devices' stored in the database.
If a match is found, MQTT sends out a 'ON' message is the average net power during the 20 sec > 0, and 'OFF' if <=0.
Also a avg power messages is sent out for the device
If the device has a stored off-time, an OFF message is queued to be sent later, along with power back to 0.
Baseline power is tracked as the minimum in the last hour and stored for 24 hours.
Baseline power is sent over MQTT every hour as the minimum power of the last 24 hours.
Unknown power is sent over MQTT every minute.

The web interface using python flask allows for analysis of saved events.
I recommend you run the code for 24 hours or longer.
Then you can do an analysis from the web interface.
It will look for similar looking events and group them in clusters.
Then you can name each cluster as a device. Multiple clusters can have the same device name.
For example, my EV charging turning on doesn't always look the same, so I have multiple devices all named 'ev'.
The off event should also have the same device name.
Refrigerators are pretty easy to recognize when they turn on, but not when they turn off.
I have an on device for them with a 15min off delay, so the system send an off message after 15 minutues.

Now that you are up and running with a number of devices, you can go tweak the 'max distance'.
Smaller Max Distance will require events to be more similar, so you can distinguish for example between two refrigerators.
It appears almost everything is a 1400W heater, not sure if I can distinguish my coffeemaker, toaster, toaster oven and teapot.

You can also browse through the past events, or type in the time of a particular event, then change the max distance until 
you have narrowed the event down to the group of matching events that identify your device and you can save the device.

Of course you can also delete devices.
