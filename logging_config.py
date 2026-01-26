#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized logging configuration 

Supports hierarchical logging configuration 
- Root logger with default level and format (Python's standard default)
- Per-module log level configuration
- Inheritance from parent loggers
- Environment variable configuration

Usage:
    from logging_config import setup_logging, get_logger
    
    # Setup logging once (typically at application start)
    setup_logging()
    
    # Get logger for your module
    logger = get_logger(__name__)
"""

import logging
import os
from typing import Optional, Dict

# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    root_level: Optional[str] = None,
    module_levels: Optional[Dict[str, str]] = None
) -> None:
    """
    Configure hierarchical logging for the application.
    
    This function is idempotent - it only configures logging once.
    Subsequent calls are ignored unless force=True.
    
    Uses Python's default logging format: %(levelname)s:%(name)s:%(message)s
    
    Args:
        root_level: Root logger level (default: INFO, or from LOG_LEVEL env var)
        module_levels: Dict mapping module names to log levels
                      Example: {'database': 'DEBUG', 'powerMonitor': 'INFO'}
    
    Environment Variables:
        LOG_LEVEL: Root logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_MODULE_LEVELS: Comma-separated list of module:level pairs
                          Example: "database:DEBUG,powerMonitor:INFO"
    """
    global _logging_configured
    
    # Only configure if not already configured
    if _logging_configured and logging.root.handlers:
        return
    
    # Get configuration from environment variables if not provided
    if root_level is None:
        root_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Parse module levels from environment if not provided
    if module_levels is None:
        module_levels_str = os.getenv('LOG_MODULE_LEVELS', '')
        if module_levels_str:
            module_levels = {}
            for pair in module_levels_str.split(','):
                if ':' in pair:
                    module, level = pair.split(':', 1)
                    module_levels[module.strip()] = level.strip().upper()
    
    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    root_log_level = level_map.get(root_level, logging.INFO)
    
    # Configure root logger using basicConfig with default format
    # Only configure if no handlers exist
    if not logging.root.handlers:
        # Use basicConfig with default format (Python's standard default)
        logging.basicConfig(
            level=root_log_level,
            # No format specified = uses default: %(levelname)s:%(name)s:%(message)s
        )
    
    # Configure per-module log levels (hierarchical)
    if module_levels:
        for module_name, level_str in module_levels.items():
            level = level_map.get(level_str.upper(), logging.INFO)
            logger = logging.getLogger(module_name)
            logger.setLevel(level)
            # Ensure logger doesn't propagate to root if we want isolation
            # (In most cases, we want propagation, so this is commented out)
            # logger.propagate = True
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module name.
    
    This is a convenience function that ensures logging is configured
    before returning the logger.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance for the module
    """
    # Auto-configure if not already done
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)


# Default configuration for this application
_DEFAULT_MODULE_LEVELS = {
    'database': 'INFO',
    'powerMonitor': 'INFO',
    'poweranalyzer': 'INFO',
    'nonsense_power_analyzer': 'INFO',
    'HassMQTT': 'INFO',
    # Suppress verbose third-party loggers
    'urllib3': 'WARNING',
    'requests': 'WARNING',
    'matplotlib': 'WARNING',
}


def setup_default_logging() -> None:
    """
    Setup logging with sensible defaults for this application.
    
    This is the recommended way to initialize logging in the application.
    Uses Python's default logging format.
    """
    setup_logging(
        root_level='INFO',
        module_levels=_DEFAULT_MODULE_LEVELS
    )


if __name__ == '__main__':
    # Test the logging configuration
    setup_default_logging()
    
    # Test different loggers
    logger_root = get_logger('root')
    logger_db = get_logger('database')
    logger_pm = get_logger('powerMonitor')
    logger_web = get_logger('nonsense_power_analyzer')
    
    logger_root.info("Root logger test")
    logger_db.info("Database logger test")
    logger_pm.info("PowerMonitor logger test")
    logger_web.info("Web app logger test")
    
    # Test different levels
    logger_db.debug("This should not appear (DEBUG level)")
    logger_db.info("This should appear (INFO level)")
    logger_db.warning("This should appear (WARNING level)")
