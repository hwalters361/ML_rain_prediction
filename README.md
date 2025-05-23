# ML_rain_prediction

This project explores the prediction of rainfall from sea surface temperature (SST) data using various machine learning models, including neural networks, random forests, and a 3D Vision Transformer (ViViT). It was developed as part of a climate data analysis project.

## Data Sources

- **Sea Surface Temperature (SST):** Sourced from NOAA's ERSST dataset.
- **Precipitation Data:** Originally obtained through collaboration; please confirm exact source with project contributors.

## Project Structure

- `Introduction.ipynb` — Preprocesses data and generates a `.npy` file; includes simple neural network model training.
- `vivit_test.ipynb` — Loads processed data and applies ViViT and Random Forest models. Also generates accuracy and confusion matrix results.
- `requirements.txt` — Lists all Python dependencies for the project.

## Setup Instructions

It's recommended to use a Python virtual environment with **Python 3.9**.

### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment
python3.9 -m venv venv

# Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Running the Code

## Run Introduction.ipynb
This processes the raw data and generates the .npy file needed for modeling. It also runs the baseline neural network model.

## Run vivit_test.ipynb
This notebook uses the processed data to train and evaluate:

 - A 3D Vision Transformer (ViViT) with early stopping and 9-class output

 - A Random Forest classifier

### Results

    Final model accuracy and confusion matrices are output in the notebooks.

    Results are based on single runs of each model.
    
