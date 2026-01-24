#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 2026

@author: Nils154

Sense UDP Listener - Listens for Sense UDP packets and prints the latest packet.
If only Sense would actually include real data in the packets,
and transmit at least once per second, could use it for the nonsense power monitor.

"""

import socket
import json
import threading
from datetime import datetime, timezone


class SenseUDPListener:
    """
    Background listener for Sense UDP/9999 packets.
    Buffers the most recent decoded packet for retrieval.
    """

    def __init__(self, port=9999):
        self.port = port
        self.latest = None
        self.latest_timestamp = None
        self._stop = threading.Event()
        self._new_packet = threading.Event()  # Signal when a new packet arrives
        self._thread = threading.Thread(target=self._listen, daemon=True)

    # TP-Link/Sense XOR decrypt
    def _tplink_decrypt(self, data):
        key = 171
        result = bytearray()
        for byte in data:
            decrypted = byte ^ key
            key = byte
            result.append(decrypted)
        return bytes(result)

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))

        while not self._stop.is_set():
            try:
                # Record receive time immediately when packet arrives (timezone-aware)
                receive_timestamp = datetime.now(timezone.utc)
                
                raw, addr = sock.recvfrom(2048)
                decrypted = self._tplink_decrypt(raw)
                decoded = json.loads(decrypted.decode("utf-8"))
                
                # UDP headers don't contain timestamps, so we use the receive time
                # Store as timezone-aware datetime object (not formatted string)
                self.latest = decoded
                self.latest_timestamp = receive_timestamp
                # Signal that a new packet has arrived
                self._new_packet.set()
            except Exception:
                continue

        sock.close()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def get_latest(self):
        return self.latest
    
    def get_latest_timestamp(self):
        """Get the timestamp of the latest packet."""
        return self.latest_timestamp


def main():
    """Test the SenseUDPListener by listening for packets and printing the latest."""
    print("Starting Sense UDP listener on port 9999...")
    print("Waiting for packets... (Press Ctrl+C to stop)\n")
    
    listener = SenseUDPListener(port=9999)
    listener.start()
    
    last_packet = None
    last_timestamp = None
    packet_count = 0
    
    # Exponential moving average for time between packets
    ema_interval = 2  # initial value for the EMA
    alpha = 0.2  # Smoothing factor (0.2 = 20% weight to new value, 80% to old average)
    
    try:
        while True:
            # Wait for a new packet with 2 second timeout
            packet_received = listener._new_packet.wait(timeout=2.0)
            
            if packet_received:
                # Clear the event and process the new packet
                listener._new_packet.clear()
                
                latest = listener.get_latest()
                latest_timestamp = listener.get_latest_timestamp()
                
                if latest is not None:
                    # Show packet if timestamp has changed, even if content is the same
                    timestamp_changed = latest_timestamp != last_timestamp
                    packet_changed = latest != last_packet
                    
                    if timestamp_changed or packet_changed:
                        packet_count += 1
                        
                        # Calculate time between packets and update EMA
                        interval_str = ""
                        if last_timestamp is not None:
                            # Calculate time difference in seconds
                            time_diff = (latest_timestamp - last_timestamp).total_seconds()
                            
                            # Update exponential moving average
                            if ema_interval is None:
                                ema_interval = time_diff
                            else:
                                ema_interval = alpha * time_diff + (1 - alpha) * ema_interval
                            
                            interval_str = f"   Interval: {time_diff:.3f}s (EMA: {ema_interval:.3f}s)"
                        
                        print(f"📦 Packet #{packet_count} received:")
                        if latest_timestamp:
                            # Format timestamp only when printing
                            print(f"   Timestamp: {latest_timestamp.isoformat()}")
                        if interval_str:
                            print(interval_str)
                        print(json.dumps(latest, indent=2))
                        print("-" * 60)
                        last_packet = latest
                        last_timestamp = latest_timestamp
            else:
                # Timeout - no packet received in 2 seconds, show we're still listening
                print(".", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n\nStopping listener...")
        listener.stop()
        print(f"Total packets received: {packet_count}")
        print("Listener stopped.")


if __name__ == "__main__":
    main()