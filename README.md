# Toyota Car Image Classification Using CNN

## Overview

This project explores image classification using a custom Convolutional Neural Network (CNN) built with PyTorch. It focuses on preparing a Toyota car image dataset, handling class imbalance, applying data augmentation, and developing a deep learning model to classify vehicle images into different Toyota model categories.

The project was developed as part of an academic assignment to practice image preprocessing, CNN architecture design, model training, and performance evaluation.

## Objectives

* Explore the distribution of images across Toyota car models.
* Clean the dataset by detecting and removing corrupted images.
* Preprocess images for deep learning.
* Address class imbalance using weighted sampling.
* Apply data augmentation to improve model generalization.
* Build and experiment with custom CNN architectures.
* Train and evaluate a model using classification metrics.

## Dataset

The project uses the Toyota car image dataset available on Kaggle.

**Dataset structure:**

```text
toyota_cars/
├── Toyota_Model_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Toyota_Model_2/
│   ├── image_001.jpg
│   └── ...
└── ...
```

Each subdirectory represents a class, and the images inside it belong to that class.

The current implementation selects the first 10 model directories alphabetically. Therefore, the classification task is limited to those selected categories rather than the entire dataset.

**Dataset source:** [Toyota Cars Over 20K Labeled Images on Kaggle](https://www.kaggle.com/)

> The dataset must be downloaded separately. It is not included in this repository.

## Technologies and Libraries

* **Python** — Programming language
* **PyTorch** — CNN implementation and model training
* **Torchvision** — Image preprocessing and dataset loading
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Pillow (PIL)** — Image validation and processing
* **Scikit-learn** — Label encoding and classification metrics
* **tqdm** — Progress monitoring

## Project Workflow

### 1. Dataset Exploration

The project identifies the available Toyota model categories, selects 10 classes, and visualizes the number of images available for each selected model.

### 2. Data Cleaning

The code checks images for corruption using Pillow and removes files that cannot be verified as valid images.

### 3. Image Preprocessing

Images are resized to **224 × 224 pixels** and converted into PyTorch tensors.

### 4. Handling Class Imbalance

The implementation calculates class frequencies and assigns inverse-frequency weights to individual samples. A `WeightedRandomSampler` is used to increase the sampling probability of images from underrepresented classes.

### 5. Data Augmentation

The augmentation pipeline includes:

* Random horizontal flipping
* Random rotation of up to 15 degrees
* Random changes to brightness, contrast, saturation, and hue
* Random resized cropping

These transformations introduce variations into the training images and can help improve generalization.

### 6. CNN Architecture

The project defines custom CNN-based architectures with configurable parameters.

The `ToyotaCNN` architecture includes:

* Three convolutional blocks
* Batch normalization
* ReLU activation
* Max pooling
* Dropout
* Adaptive average pooling
* A fully connected classification layer

The script also experiments with dense layers and defines `ToyotaCNN_WithDense` and `ToyotaCNN_Final` as additional architectural variants.

### 7. Model Training

The training code includes:

* Adam optimizer
* Cross-entropy loss
* Training and validation loops
* Best-model weight tracking
* Early-stopping logic
* A learning-rate scheduler using `ReduceLROnPlateau`

The script defines a maximum of five training epochs in its current configuration.

### 8. Model Evaluation

The code includes the following evaluation metrics:

* Accuracy
* Macro-averaged precision
* Macro-averaged recall
* Macro-averaged F1-score
* Confusion matrix

These metrics provide different perspectives on classification performance, particularly when class distributions are imbalanced.

## Installation

Clone the repository:

```bash
git clone https://github.com/GhazalCh/toyota-car-image-classification.git
cd toyota-car-image-classification
```

Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow tqdm
```

**Note:** Install a PyTorch version compatible with your operating system and hardware. GPU support is optional.

## Dataset Configuration

The original script uses Kaggle-specific paths. Update the dataset paths before running the code in another environment.

For example:

```python
data_dir = "/path/to/toyota_cars"

src_root = "/path/to/toyota_image_dataset_v2"

dst_root = "/path/to/working/toyota_cleaned_dataset"
```

The paths used for dataset loading and cleaning must point to the correct directories in your environment.

The script should also be reviewed to ensure that the dataset directory structure is consistent across all preprocessing stages.

## Running the Project

After installing the dependencies and configuring the dataset paths, run:

```bash
python Assignment1.py
```

The script is designed to execute in a Python environment with the required dependencies and dataset available.

Some sections may require corrections before the complete training and evaluation pipeline can run successfully.

## Evaluation Metrics

| Metric            | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| Accuracy          | Overall proportion of correct predictions                            |
| Precision (Macro) | Average precision across classes, giving each class equal weight     |
| Recall (Macro)    | Average recall across classes, giving each class equal weight        |
| F1-score (Macro)  | Average F1-score across classes                                      |
| Confusion Matrix  | Shows the distribution of correct and incorrect predictions by class |

Actual evaluation results should be added after the training and evaluation pipeline runs successfully.

## Current Limitations

* The implementation selects 10 classes based on alphabetical directory order.
* Dataset paths are specific to the original Kaggle environment and must be configured for local execution.
* Multiple preprocessing and dataset-loading sections are present in the script.
* The training and evaluation pipeline requires further debugging and integration.
* The additional dense architecture needs to be checked for consistency between its feature dimensions, forward pass, and output layer.
* The current script does not provide verified final evaluation results.

## Future Improvements

* Refactor the notebook-style code into reusable modules.
* Use a consistent dataset preparation pipeline.
* Separate training, validation, and test datasets.
* Ensure that data augmentation is applied only to the training set.
* Integrate class balancing into the final training pipeline.
* Add model checkpoint saving and loading.
* Track training and validation loss and metrics across epochs.
* Compare the custom CNN with transfer-learning models such as ResNet.
* Add reproducibility controls by setting random seeds.
* Save evaluation plots and classification reports.

## Learning Outcomes

This project provides practical experience with:

* Image classification using deep learning
* Custom CNN architecture design
* Image preprocessing and data augmentation
* Class imbalance handling
* Model training and validation
* Classification metrics and confusion matrix visualization

## License

This project was developed for educational purposes. Add a license if you intend to distribute the code for reuse.
