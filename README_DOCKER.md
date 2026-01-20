# Docker Setup for Power Monitor

This Docker setup runs both `powerMonitor.py` and `nonsense_power_analyzer.py` in a single container.

## Prerequisites

- Docker and Docker Compose installed

## Environment Variables

MQTT variables to send the data
ENPHASE account info
ENVOY_IP
ENVOY_SERIAL can be found in envoy app->menu->system->devices->gateway->SN)
check the log after running the first time for the EID values.
depending on your setup you need either both EIDs or only the GRID_EID.

```
MQTT_HOST='your_mqtt_host'
MQTT_PORT=1883
MQTT_USER='your_mqtt_user'
MQTT_PASSWORD='your_mqtt_password'

ENPHASE_EMAIL="your_email"
ENPHASE_PASSWORD="your_password"
ENVOY_IP="your_envoy_ip"
ENVOY_SERIAL="your_envoy_serial"

GRID_EID=704643584
SOLAR_EID=704643328
```

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
