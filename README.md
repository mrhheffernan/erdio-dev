# Erdio

By Matthew Frick, Matthew Heffernan, and Paul Jredini

Erdos Institute Spring 2022 Bootcamp Capstone Project

### Executive Summary

Timely identification of safety-critical events, such as gunshots, is of great importance to public safety stakeholders. However, existing systems only deliver limited value by not classifying additional urban sounds. We perform classification of environmental sounds to detect safety-critical events, in particular gunshots, and provide information on first-response via siren detection. We also engineer general features for off-line classification tasks and demonstrate how this system can provide value to additional stakeholders in the film and television industry. 

The primary data source is the UrbanSound8K dataset.

Citation: 
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

### Additional Materials:

Presentation:https://docs.google.com/presentation/d/e/2PACX-1vRw9xHJe7y-EH16iL0kLPknPa5RUWiXMjb64CZlqCGdaaoxFanfU9NzWXLKFb9G7QsuvxqMSSqWgOPF/pub?start=false&loop=false&delayms=10000

## Setup

To run the notebook, download and extract the data from https://zenodo.org/record/1203745 to "large_data/" directory. 6G of disk space is required for this download.

The following packages are required:
- librosa
- numpy
- sklearn
- keras
- tensorflow
- matplotlib
- seaborn
- pandas


## Data exploration

Gunshot detection in the UrbanSound8K dataset: Basic visualization notebook includes data loading and various visualizations that might be useful, including some often used in audio analysis.

## Feature Extraction

## Feature Inspection

## Classification


