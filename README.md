# Audio Classification Using Mel Spectrograms

The files in this repo provide a basic introduction to classification of audio data using Mel spectrograms to train a Convolutional Neural Network (CNN). Data for this project were acquired from the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). Code for this project is included in the notebook file. Prior to running the notebook, you will need to download the UrbanSound8K dataset and save the downloaded directory in your project folder. 

It is highly recommended that this notebook be run using a GPU for model training. Please note that the notebook generates a numpy array for each audio file representing its Mel spectrogram and saves the array as an NPZ file for processing later. These files, along with the raw audio files from UrbanSound8K, are not included in this repo to conserve space.

I also wrote a Medium article outlining the process used to complete this project. Please give it some claps if you find it helpful. 
