# Erdio

By Matthew Frick, Matthew Heffernan, and Paul Jredini

Erdos Institute Spring 2022 Bootcamp Capstone Project

### Executive Summary

Timely identification of safety-critical events, such as gunshots, is of great importance to public safety stakeholders. However, existing systems only deliver limited value by not classifying additional urban sounds. We perform classification of environmental sounds to detect safety-critical events, in particular gunshots, and provide information on first-response via siren detection. We also engineer general features for off-line classification tasks and demonstrate how this system can provide value to additional stakeholders in the film and television industry. 

Our key performance indicator (KPI) is the F measure on the gunshots specifically, and then across all classes.

The primary data source is the UrbanSound8K dataset.

Citation: 
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

### Additional Materials:

Presentation:https://docs.google.com/presentation/d/e/2PACX-1vRw9xHJe7y-EH16iL0kLPknPa5RUWiXMjb64CZlqCGdaaoxFanfU9NzWXLKFb9G7QsuvxqMSSqWgOPF/pub?start=false&loop=false&delayms=10000

## Setup

To run the notebook, download and extract the data from https://zenodo.org/record/1203745 to "large_data/" directory. 6G of disk space is required for this download.

This enitre project was written in python 3.6+. Users with earlier versions of python may experience incompatibilities. 

The following packages are required:
- jupyter 
- librosa
- numpy
- sklearn
- keras
- tensorflow
- matplotlib
- seaborn
- pandas

These can be installed via ```pip packagename``` or ```conda packagename```

## Walkthrough

The notebooks are designed to be self-contained. However, feature extraction is presently computationally-intensive and required the use of high-performance computing resources. As such, we recommend the interested user focus on ```Classification.ipynb```, the classification notebook.

The classification notebook can be run start-to-finish out of the box as a Jupyter Notebook.

## Data exploration

Gunshot detection in the UrbanSound8K dataset: Basic visualization notebook includes data loading and various visualizations that might be useful, including some often used in audio analysis.

## Feature Extraction

## Feature Inspection

## Classification

In this notebook, we test a wide variety of classifiers to determine their ability to differentiate between the various sounds. We test classifiers ranging from simple (logistic regression) to complex (boosted random forest, stacked classifiers) to ensure that the final model complexity is justified by the data.

We find good success in our KPIs, with an F of over 85% for the logistic regression, our top-performing model. Using a Random Forest for feature selection, we find the following performance in 10-fold cross-validation.

| Classifier   | Gunshot Recall | Gunshot Fmeasure | Cross-class Fmeasure | 
|--------------|----------------|----------------------|----------------------|
| Logistic Regression | **0.8819 +/- 0.0334**  | **0.8706 +/- 0.0204**  | **0.607 +/- 0.007** |
| Nearest Neighbors   | 0.5065 +/- 0.0483  | 0.5662 +/- 0.0455  | 0.4881 +/- 0.015 |
| Decision Tree       | 0.6269 +/- 0.0618  | 0.6636 +/- 0.0498  | 0.4402 +/- 0.014 |
| Random Forest       | 0.8038 +/- 0.0544  | 0.8285 +/- 0.039  | 0.5938 +/- 0.011 |
| Neural Net          | 0.7951 +/- 0.0445  | 0.7955 +/- 0.0313  | 0.6019 +/- 0.014 |
| AdaBoost            | 0.7131 +/- 0.0441  | 0.5456 +/- 0.0354  | 0.3599 +/- 0.011 |
| Naive Bayes         | 0.5657 +/- 0.0363  | 0.5515 +/- 0.0418  | nan +/- nan |
| Quadratic Discriminant Analysis                 | 0.5674 +/- 0.0529  | 0.7042 +/- 0.0458  | nan +/- nan |
| Linear Discriminant Analysis                 | 0.7964 +/- 0.0384  | 0.7563 +/- 0.0376  | 0.5978 +/- 0.012 |
| ExtraTrees          | 0.774 +/- 0.0507   | 0.8154 +/- 0.037  | 0.57 +/- 0.014 |
| Boosted Random Forest          | **0.8182 +/- 0.0539**  | **0.8392 +/- 0.0377**  | **0.6107 +/- 0.014** |
| Stacking            | **0.8666 +/- 0.0371**  | **0.8451 +/- 0.0344**  | **0.6469 +/- 0.011** |

The most successful models, the logistic regression and the Boosted Random Forest, are finally trained on all models to produce our final models.

Error bars come from estimating the standard error across 10 cross validation folds and show that the top three models, highlighted in bold, are consistent with each other for our KPIs. We deploy both the logistic regression and the boosted random forest in a test case to investigate outside-of-set performance.

## Field testing

Field testing was performed on an episode of Futurama with known gunshot noise, demonstrating its value to stakeholders in the film and television industry.
