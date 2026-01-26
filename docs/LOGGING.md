# Logging Configuration

This application uses a hierarchical logging system similar to Home Assistant, allowing per-module log level configuration with inheritance.

## Overview

The logging system is centralized in `logging_config.py` and provides:

- **Hierarchical loggers**: Child loggers inherit from parent loggers
- **Per-module configuration**: Set different log levels for different modules
- **Local timezone timestamps**: Logs show timestamps in your local timezone
- **Environment variable configuration**: Configure via environment variables
- **Idempotent setup**: Safe to call `setup_logging()` multiple times

## Basic Usage

### In Your Module

```python
from logging_config import setup_default_logging, get_logger

# Setup logging once (typically at application start)
setup_default_logging()

# Get logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("This is an info message")
logger.debug("This is a debug message")
logger.warning("This is a warning")
```

## Logger Hierarchy

Loggers are organized in a tree structure based on their names (using dots as separators):

```
root (INFO)
├── database (INFO)
├── powerMonitor (INFO)
├── poweranalyzer (INFO)
├── nonsense_power_analyzer (INFO)
└── HassMQTT (INFO)
```

Child loggers inherit the log level from their parent unless explicitly set.

## Configuration Methods

### 1. Default Configuration (Recommended)

```python
from logging_config import setup_default_logging
setup_default_logging()
```

This sets up logging with sensible defaults:
- Root level: `INFO`
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Local timezone timestamps
- Per-module levels configured

### 2. Custom Configuration

```python
from logging_config import setup_logging

setup_logging(
    root_level='WARNING',  # Root logger level
    default_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    module_levels={
        'database': 'DEBUG',
        'powerMonitor': 'INFO',
        'poweranalyzer': 'WARNING',
    },
    use_local_time=True
)
```

### 3. Environment Variables

You can configure logging via environment variables:

```bash
# Root logger level
export LOG_LEVEL=DEBUG

# Custom format (optional)
export LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Per-module levels (comma-separated)
export LOG_MODULE_LEVELS="database:DEBUG,powerMonitor:INFO,poweranalyzer:WARNING"
```

### 4. Runtime Configuration

Change log levels at runtime:

```python
from logging_config import set_module_level

# Set database logger to DEBUG
set_module_level('database', 'DEBUG')

# Set powerMonitor logger to WARNING
set_module_level('powerMonitor', 'WARNING')
```

## Log Levels

Available log levels (from most verbose to least):

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: An indication that something unexpected happened
- `ERROR`: A more serious problem, the software is still able to function
- `CRITICAL`: A serious error, the program itself may be unable to continue

## Format Placeholders

Common format placeholders:

| Placeholder | Description | Example |
|------------|-------------|---------|
| `%(asctime)s` | Human-readable timestamp (local time) | `2024-01-15 14:30:45 CST` |
| `%(name)s` | Logger name (module name) | `database` |
| `%(levelname)s` | Log level | `INFO` |
| `%(message)s` | Log message | `Loaded 100 events` |
| `%(filename)s` | Source filename | `database.py` |
| `%(lineno)d` | Line number | `42` |
| `%(funcName)s` | Function name | `load_events` |

## Example Output

With default configuration:

```
2024-01-15 14:30:45 CST - database - INFO - Loaded 100 events from database
2024-01-15 14:30:46 CST - powerMonitor - INFO - Event triggered
2024-01-15 14:30:47 CST - nonsense_power_analyzer - INFO - Plot generated successfully
```

## How It Works

1. **First Import**: When any module imports `logging_config`, it can call `setup_logging()` or `setup_default_logging()`
2. **Idempotent**: If logging is already configured, subsequent calls are ignored (unless you force it)
3. **Inheritance**: Child loggers inherit log levels from parent loggers
4. **Propagation**: By default, log messages propagate up the logger hierarchy to the root logger

## Comparison with Home Assistant

This logging system is inspired by Home Assistant's logging configuration:

- ✅ Hierarchical logger structure
- ✅ Per-module log level configuration
- ✅ Environment variable support
- ✅ Local timezone timestamps
- ✅ Centralized configuration

## Troubleshooting

### Logs show UTC instead of local time

Make sure `use_local_time=True` is set (it's the default in `setup_default_logging()`).

### Too many/few log messages

Adjust the log levels:
- More verbose: Set module level to `DEBUG`
- Less verbose: Set module level to `WARNING` or `ERROR`

### Logs from third-party libraries are too verbose

The default configuration suppresses verbose third-party loggers:
- `urllib3`: `WARNING`
- `requests`: `WARNING`
- `matplotlib`: `WARNING`

You can add more in the `module_levels` dictionary.

## See Also

- Python logging documentation: https://docs.python.org/3/library/logging.html
- Home Assistant logging: https://www.home-assistant.io/integrations/logger/
