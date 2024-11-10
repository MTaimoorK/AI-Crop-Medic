# Crop Disease Detection

This project leverages machine learning and image processing to detect diseases in wheat, rice, cotton, and sugarcane leaves, helping farmers and agronomists with timely diagnosis and intervention to maintain crop health and improve yield.

## Features

- **Four ML Models**: Separate models for detecting diseases in wheat, rice, cotton, and sugarcane leaves, each stored in a `.pkl` file.
- **Flask API with HTML Template**: A Flask application that serves as an API endpoint for uploading leaf images and receiving disease predictions with an HTML template for easy web access.

## Project Structure

- **models/**
  - `wheat_model.pkl` - Trained model for wheat leaf disease detection.
  - `rice_model.pkl` - Trained model for rice leaf disease detection.
  - `cotton_model.pkl` - Trained model for cotton leaf disease detection.
  - `sugarcane_model.pkl` - Trained model for sugarcane leaf disease detection.

- **templates/**
  - `index.html` - HTML template for uploading images and selecting crop type.

- **app.py** - Flask app to handle image uploads, display HTML form, and return predictions.

- **requirements.txt** - List of dependencies needed to run the app.

## Prerequisites

- Python 3.x
- Install dependencies:
  ```bash
  pip install -r requirements.txt

