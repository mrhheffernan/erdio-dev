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

### Data Cleaning

Much of the audio data in the UrbanSound8K database has a significant amount of background or "room" noise. In an attempt to clean up this background we design a filter pipeline. The first step of the process is to take the hilbert transform of the time domain waveform. The absolute value of this hilbert transform is then the instantaneous amplitude of the waveform. This instantaneous amplitude is processed by a low pass filter. The second step is to find the root mean square of the waveform. The raw waveform is then scaled by the filtered instantaneous amplitude divided by the root mean square. Finally, an overall factor is added to the waveform such that the maximum amplitude is maintained between the raw and the filtered waveforms. If the waveform has a large peak this filter thus biases the waveform towards keeping the large swings, while minimizing the "white noise" styled hum. Comparatively, if the waveform does not have large swings this filter does very little to the waveform as the root mean square is everywhere close to the instantaneous amplitude.

### Musically Informed Feature Selection

In an attempt to select features that are understandable to humans we look to certain ideas within music theory.

#### The Equalizer Scale

An additional issue with the audio data in the UrbanSound8K database is that the duration of the audio samples are non-uniform. In particular, the audio files labelled as "gun_shot" are significantly shorted in duration than many of the other class labels. As such we aim to extract features in the frequency domain. Normally, the total duration of a signal determines the lowest possible frequency in the discrete fourier transform. As such the first set of features we extract is a binned version of the fourier transform, where the bins are split into a set of human audible ranges; the bass, the midrange, the high end, the sub-audible (infrasound), and supra-audible (ultrasound). By taking the mean power in each of the bins we then remove any biasing information from the audio duration from our features.

#### Harmonics and Percussives

The librosa package has a convenient function that is able to separate a waveform into its harmonic and percussive components. In order to extract further musically informed features we utilize this decomposition and extract the amount of power in the harmonic and percussive components, as well as the ratio of these powers.

The harmonic component of this waveform is then short-time fourier transformed and the frequencies of each time slice are binned according to the MIDI scale. This scale is a simple integer representation of the keys of a piano that has been tuned using an equal temperament. After this binning has been performed it is relatively simple to extract whether a number of musically common interval pairs and chord triplets are present. The total number of major thirds, minor thirds, perfect fifths, and major chords above a power cutoff are then counted across all time slices and extracted as a feature.

The percussive component is somewhat simpler; we exploit the onset strength functions from librosa to determine the total number of large percussive amplitude onsets in the percussive component of the waveform.

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
