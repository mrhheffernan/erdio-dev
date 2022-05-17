# Data exploration

Consider keeping large data files in a directory named "large_data/" in your cloned repo directory, as git is now set up to ignore this directory, avoiding pushing the data to github.

## Gunshot detection in UrbanSound8K dataset

Basic visualization notebook includes data loading and various visualizations that might be useful, including some often used in audio analysis.
To run the notebook, download and extract the data from https://zenodo.org/record/1203745 to "large_data/" directory.

It's yet unclear if the commonly used transforms available in librosa package are useful for gunshots, as they seem to be aimed more towards tonal voice and music analysis.
