# Nonsense Power Analyzer - Web Application

A web-based interface for power event analysis, clustering, and device identification.

## Features

- **Data Management**: data stored in an sqlite database
- **Analysis**: Run fresh clustering analysis or load previous results
- **Device Management**: View and manage identified devices
- **Real-time Status**: Monitor events

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_web.txt
```

2. Ensure you have the `data/` directory 

## Running the Application

```bash
python nonsense_power_analyzer.py
```

The application will start on `http://localhost:8888`

## Architecture

The web application is built on Flask and uses the `PowerEventAnalyzer` class from `powerAnalyzerv4.py` for all analysis operations. The frontend uses vanilla JavaScript for interactivity.

