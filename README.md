# DE-Chem: Deep Ensemble Chemical Kinetics Modeling

## Overview
DE-Chem is a deep learning project focused on modeling chemical kinetics using deep ensemble neural networks. This repository contains code for data preprocessing, training, evaluation, and visualization of models designed for dynamic chemical systems analysis with mass conservation and uncertainty quantification.

## Features
- Data import and preprocessing pipelines optimized for chemical kinetics data
- Deep ensemble model architecture implemented with TensorFlow
- Custom hybrid loss functions incorporating mass conservation constraints
- Comprehensive model evaluation metrics including uncertainty quantification
- Visualization utilities for model predictions, species comparisons, and uncertainty growth
- Modular and extensible Python codebase for experimentation and development

## Repository Structure
- `data_preprocessing_L.py`: Data loading, scaling, and input-output pair generation
- `deep_ensemble_L.py`: Deep ensemble model implementation
- `train_L.py`: Training routines for the models with various loss settings
- `evaluate_L.py`, `eval_function.py`, `compare_metrics.py`: Evaluation functions and metric calculations
- `visualize_L.py`: Plotting and visualization utilities for analysis of results
- `main_L.py`: Main script to run training and evaluation pipelines integrating all components
- `base_model_L.py`: Base neural network model class providing shared utilities
- `data_analysis.py`: Exploratory data analysis tools and scripts

## Requirements
- Python 3.10+
- TensorFlow (version corresponding to project compatibility)
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SciPy

Please refer to `requirements.txt` or `environment.yml` for detailed dependencies and installation instructions.

## Usage

### Setup
1. Clone the repository:

git clone https://github.com/cihatemreustun/DE-Chem.git

2. Install dependencies using Conda or pip.

### Running Training and Evaluation
Run the main training and evaluation pipeline:

python main_L.py

Parameters such as learning rate, number of models in the ensemble, mass constraint usage, and phases of training can be configured directly in `main_L.py` or through modifying functions in `train_L.py`.

### Visualization
Use `visualize_L.py` to generate plots for training history, species predictions, uncertainty growth, and mass conservation. Example commands are shown in the script.

## Development
- Follow the modular structure to add new models or loss functions.
- Extend evaluation metrics by modifying `eval_function.py` and `compare_metrics.py`.
- Use `data_analysis.py` for dataset insights and preprocessing validation.

## Contribution
Contributions are welcome! Please fork the repo, add features or fixes, and submit pull requests.

## License
Specify the license under which the project is shared here (e.g., MIT License).

## Contact
For questions or collaborations, please contact Emre at [prw287@qmul.ac.uk].


