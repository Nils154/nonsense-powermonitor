# Docker Setup for Power Monitor

This Docker setup runs both `powerMonitor.py` and `nonsense_power_analyzer.py` in a single container.

## Prerequisites

- Docker and Docker Compose installed

## Environment Variables

**⚠️ IMPORTANT: You must configure environment variables before starting the container.**

The container requires several environment variables to function. You can configure them in one of two ways:

### Option 1: Using a .env file (Recommended)

1. Create a `.env` file in the project root directory
2. Add your configuration values:

```
MQTT_HOST=your_mqtt_host
MQTT_PORT=1883
MQTT_USER=your_mqtt_user
MQTT_PASSWORD=your_mqtt_password

ENPHASE_EMAIL=your_email
ENPHASE_PASSWORD=your_password
ENVOY_IP=your_envoy_ip
ENVOY_SERIAL=your_envoy_serial

GRID_EID=123456789
SOLAR_EID=1234567890
```

3. The `docker-compose.yml` is already configured to automatically load the `.env` file.

### Option 2: Uncomment variables in docker-compose.yml

1. Open `docker-compose.yml`
2. Uncomment the environment variables under the `environment:` section
3. Replace the placeholder values with your actual configuration:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - MQTT_HOST=mqtthost.local
  - MQTT_PORT=1883
  - MQTT_USER=you
  - MQTT_PASSWORD=secret
  - ENPHASE_EMAIL=you@gmail.com
  - ENPHASE_PASSWORD=alsosecret
  - ENVOY_IP=192.168.1.36
  - ENVOY_SERIAL=12345678910
  - GRID_EID=123456789
  - SOLAR_EID=1234567890
```

### Required Variables

The following variables are **required** for the container to start:

- `ENPHASE_EMAIL` - Your Enphase account email
- `ENPHASE_PASSWORD` - Your Enphase account password
- `ENVOY_IP` - Local IP address of your Envoy device
- `ENVOY_SERIAL` - Envoy serial number (can be found in envoy app->menu->system->devices->gateway->SN)
- `MQTT_HOST` - MQTT broker hostname or IP address (required if using MQTT). **Note: Local DNS resolution (e.g., `.local` hostnames) does not work in Docker containers. Use an IP address instead of a hostname if you encounter DNS resolution errors.**

### Optional Variables

The following variables are optional:

- `GRID_EID` - Grid/consumption meter eid (integer). If not provided, the first meter found will be used. Check the log after running the first time for the EID values.
- `SOLAR_EID` - Solar meter eid (integer). Only needed if you have solar behind the grid meter. Depending on your setup you may need both EIDs or only the GRID_EID.
- `MQTT_PORT` - MQTT broker port (default: 1883)
- `MQTT_USER` - MQTT username (optional, for authenticated MQTT)
- `MQTT_PASSWORD` - MQTT password (optional, for authenticated MQTT)

## Building and Running

### Using Docker Compose (Recommended)

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t nils154/nonsense-powermonitor .

# Run the container
docker run -d \
  --name powermonitor \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/plots:/app/plots \
  -v $(pwd)/.env:/app/.env:ro \
  nils154/nonsense-powermonitor

# View logs
docker logs -f powermonitor

# Stop
docker stop powermonitor
docker rm powermonitor
```

## Accessing the Application

Once running, access the web interface at:
- http://localhost:8888

## Data Persistence

The following directories are mounted as volumes to persist data:
- `./data` - Contains the SQLite database and analysis files
- `./plots` - Contains generated plot images

## Troubleshooting

### DNS Resolution Issues

**⚠️ Local DNS resolution does not work in Docker containers.**

If you encounter `socket.gaierror: [Errno -2] Name or service not known` errors, this is likely because:
- You're using a hostname (like `mqtthost.local`) that cannot be resolved inside the container
- Local DNS (mDNS/Bonjour) services are not available to Docker containers

**Solution:** Use IP addresses instead of hostnames for `MQTT_HOST` and `ENVOY_IP`:
- Instead of: `MQTT_HOST=mqtthost.local`
- Use: `MQTT_HOST=192.168.1.100` (replace with actual IP address)

### Check if both processes are running
```bash
docker exec powermonitor ps aux
```

### View logs
```bash
docker compose logs -f
# or
docker logs -f powermonitor
```

### Access container shell
```bash
docker exec -it powermonitor /bin/bash
```

### Rebuild after code changes
```bash
docker compose build --no-cache
docker compose up -d
```
