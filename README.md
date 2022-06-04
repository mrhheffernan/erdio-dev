# Erdio

By Matthew Frick, Matthew Heffernan, and Paul Jredini

Erdos Institute Spring 2022 Bootcamp Capstone Project

### Executive Summary

Timely identification of safety-critical events, such as gunshots, is of great importance to public safety stakeholders. However, existing systems only deliver limited value by not classifying additional urban sounds. We perform classification of environmental sounds to detect safety-critical events, in particular gunshots, and provide information on first-response via siren detection. We also engineer general features for off-line classification tasks and demonstrate how this system can provide value to additional stakeholders in the film and television industry. 

Our key performance indicator (KPI) is first and foremost the F measure on the gunshot class detection, and then across all other labeled classes.



### Additional Materials:

Presentation: [Here](https://docs.google.com/presentation/d/e/2PACX-1vRw9xHJe7y-EH16iL0kLPknPa5RUWiXMjb64CZlqCGdaaoxFanfU9NzWXLKFb9G7QsuvxqMSSqWgOPF/pub?start=false&loop=false&delayms=10000)

## Setup

The primary data source is the UrbanSound8K dataset. Citation for the dataset: 
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

The raw dataset used in this project is too large to upload to Github; users who wish to run all the notebooks thus need to download and extract the data from https://zenodo.org/record/1203745 to "large_data/" directory. 6G of disk space is required for this download. However, some notebooks do not require the dataset to function, as we include extracted features as CSV files.

This entire project was written in python 3.6+. Users with earlier versions of python may experience incompatibilities. 

The following packages are required:
- jupyter 
- librosa
- numpy (if encountering dependencies issues, install specifically version 1.21.4)
- pandas
- sklearn
- keras
- tensorflow
- matplotlib
- seaborn

These can be installed via ```pip packagename``` or ```conda packagename```

## Walkthrough

The notebooks are designed to demonstrate our approach from start to finish, and should be followed in this order:

- [data_exploration.ipynb](./data_exploration.ipynb)
- [feature_extraction.ipynb](./feature_extraction.ipynb)
- [feature_inspection.ipynb](./feature_inspection.ipynb)
- [Classification.ipynb](./Classification.ipynb)
- [classification_futurama.ipynb](./classification_futurama.ipynb)

However, in practice, running all the notebooks is likely to be computationally intensive, especially for part of the feature generation procedure which required high-performance computing resources. Additionally, the data used in the final notebook could not be made directly available due to copyright issues. In light of these facts, we have included CSV files of extracted features, for both the UrbanSound8K dataset and the dataset used in the final notebook, to allow users to more easily follow our procedure if desired.

We describe the general content of each notebook here. More details can be found interspersed in the relevant notebooks.

### [Data exploration](./data_exploration.ipynb)

The UrbanSound8K dataset consists of 8732 single-channel audio files, each labeled as one of 10 classes including the "gun_shot" class. We demonstrate how to load the data and necessary metadata. We then examine the distribution of classes in the dataset. We also showcase some low-quality or badly labeled audio samples, anticipating possible issues in their classification.

### [Feature Extraction](./feature_extraction.ipynb)

#### Data Cleaning

Much of the audio data in the UrbanSound8K database has a significant amount of background noise. In an attempt to clean up this background, we design a filter pipeline consisting of a so-called Hilbert transform whose goal is to amplify the large "swings" of the audio waveforms while suppressing the constant "hum" of white noise in the background. This transform is applied before the extraction of most, but not all, the features summarized below.

#### Musically Informed Feature Selection

In an attempt to select features that are understandable to humans we look to certain ideas within music theory. Additionally, an issue with the audio data in the UrbanSound8K database is that the duration of the audio samples are non-uniform. In particular, the audio files labelled as "gun_shot" are significantly shorter in duration than many of the other class labels. As such, most of the features we extract are from the frequency domain.

##### The Equalizer Scale

The first set of features is a binned version of the Fourier transform, where the bins are split into a set of human audible ranges; the bass, the midrange, the high end, the sub-audible (infrasound), and supra-audible (ultrasound). By taking the mean power in each of the bins we then remove biasing information from the audio duration from our features.

##### Harmonics and Percussives

We decompose the audio samples into harmonic and percussive components, and extract the amount of power in each, as well as the ratio of these powers. The harmonic component is also used to extract chord information in the MIDI musical scale, while the percussive component is used to obtain a count of percussive hits present in the sample.

##### Fundamental frequency signal

To diversify our musically-informed feature set, we exploit Librosa's fundamental frequency estimation algorithm to obtain an effective "signal" for the presence of a clear tone indicative of music or similar sounds.

#### Spectral Features in Max Power Window

By using the rolling average of spectral power for each audio sample, we locate the time window containing the maximal amount of power in each sample. This isolates what is likely the most pertinent part of the audio, especially for loud sounds such as those of the "gun_shot" class. We then extract from this window various spectral features commonly used in audio analysis, such as spectral flatness and rolloff, and calculate their descriptive statistical values such as median, IQR, etc.

### [Feature Inspection](./feature_inspection.ipynb)

We demonstrate how some of the features we engineered from the raw dataset discriminate between classes, often in an intuitively understandable manner, before we even attempt to input them into machine learning models.

### [Classification](./Classification.ipynb)

In this notebook, we test a wide variety of classifiers to determine their ability to differentiate between the various sounds. The classifiers range from simple (logistic regression) to complex (boosted random forest, stacked classifiers) to ensure that the final model complexity is justified by the data.

We prune down our extracted features to only the most relevant using a Random Forest model's estimation of feature importance.

We find good success in our KPIs, with an F measure of over 85% for the logistic regression, our top-performing model. We find the following performance in 10-fold cross-validation.

| Classifier   | Gunshot Recall | Gunshot Fmeasure | Cross-class Fmeasure | 
|--------------|----------------|----------------------|----------------------|
| Logistic Regression | **0.8819 +/- 0.0334**  | **0.8706 +/- 0.0204**  | **0.607 +/- 0.007** |
| Nearest Neighbors   | 0.5065 +/- 0.0483  | 0.5662 +/- 0.0455  | 0.4881 +/- 0.015 |
| Decision Tree       | 0.6269 +/- 0.0618  | 0.6636 +/- 0.0498  | 0.4402 +/- 0.014 |
| Random Forest       | 0.8038 +/- 0.0544  | 0.8285 +/- 0.039  | 0.5938 +/- 0.011 |
| Neural Net          | 0.7951 +/- 0.0445  | 0.7955 +/- 0.0313  | 0.6019 +/- 0.014 |
| AdaBoost            | 0.7131 +/- 0.0441  | 0.5456 +/- 0.0354  | 0.3599 +/- 0.011 |
| Linear Discriminant Analysis                 | 0.7964 +/- 0.0384  | 0.7563 +/- 0.0376  | 0.5978 +/- 0.012 |
| ExtraTrees          | 0.774 +/- 0.0507   | 0.8154 +/- 0.037  | 0.57 +/- 0.014 |
| Boosted Random Forest          | **0.8182 +/- 0.0539**  | **0.8392 +/- 0.0377**  | **0.6107 +/- 0.014** |
| Stacking            | **0.8666 +/- 0.0371**  | **0.8451 +/- 0.0344**  | **0.6469 +/- 0.011** |

The most successful models, the logistic regression and the Boosted Random Forest, are finally trained on all folds to produce our final models.

Error bars come from estimating the standard error across 10 cross validation folds and show that the top three models, highlighted in bold, are consistent with each other for our KPIs.

### [Field testing on Futurama episode](./classification_futurama.ipynb)

We deploy both the logistic regression and the boosted random forest in a test case to investigate outside-of-set performance, demonstrating its value to stakeholders in the film and television industry. The dataset consists of an episode of Futurama known to contain several gunshot sounds, split into 5 second audio fragments. In our presentation slides, we further showcase the ability of the models to find the gunshot sounds despite some confusion with similar sudden sounds present in the episode.
