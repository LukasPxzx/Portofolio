# Real Estate WebApp with Time-Series Mold Prediction

## Overview

This project is a Real Estate WebApp that integrates IoT sensor data for predicting mold growth in real estate properties. It features real-time updates and user authentication, utilizing a Flask backend, SQLite database, and TensorFlow.js for AI model deployment.

## Features

- **User Authentication**: Secure login for users.
- **Real-Time Data**: Integrates IoT sensor data for temperature, humidity, luminosity, and HCHO levels.
- **Mold Growth Prediction**: Uses trained models to predict mold growth based on environmental factors.
- **Interactive UI**: Built with HTML, CSS, and JavaScript, providing a responsive user experience.

## Installation

### Prerequisites

- Python 3.7 or higher no higher than 3.9.20 (Recommended)
- SQLite
- Node.js (for TensorFlow.js)

### Required Libraries

Install the necessary Python packages:

```bash
pip install flask flask-socketio tensorflow numpy pandas matplotlib scikit-learn tensorflowjs smbus2
```

### Additional Libraries

The following libraries are used in the project but may not require separate installation if already included:

- `hashlib`
- `logging`
- `json`
- `os`
- `shutil`
- `scipy`

### Setting Up the Database

1. Clone the repository or download the files.
2. Navigate to the project directory.
3. Run the following command to create the database and tables:

    ```bash
    python -c "from database import create_tables; create_tables()"
    ```

4. Insert static data into the database:

    ```bash
    python -c "from database import insert_static_data; insert_static_data()"
    ```

## Hardware Requirements

- IoT sensors:
  - **gen_7700**: For measuring HCHO levels.
  - **gen_sht4x**: For measuring temperature and humidity.
  - **gen_sfa30**: For measuring CO2 levels.
- A Raspberry Pi or any other compatible device for sensor integration.

## Running the Application

1. Start the Flask server:

    ```bash
    python main.py
    ```

2. Open your web browser and navigate to `http://localhost:8000`.

## Usage

- **Login**: Enter your username, password and verify your face to access the application.
- **Data Visualization**: View real-time sensor data and predictions.
- **Property Management**: Add, update, and manage properties through the UI.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or enhancements.

## Acknowledgments

- TensorFlow.js for machine learning model deployment.
- Flask for web server management.
- SQLite for lightweight database management.
- Teachable Machine for Face Recognition Model
