# Audio Classification Using Mel Spectrograms

The files in this repo provide a basic introduction to classification of audio data using Mel spectrograms to train a Convolutional Neural Network (CNN). Data for this project were acquired from the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). Code for this project is included in the notebook file. It is also split into individual scripts for generation of example spectrograms, file processing, and model training. Prior to running the notebook or scripts, you will need to download the UrbanSound8K dataset and save the downloaded directory in your project folder. The dataset can be found at the link above. Alternatively, you can [download it from Kaggle](https://www.kaggle.com/chrisfilo/urbansound8k).  

It is highly recommended that the notebook and/or scripts be run using a GPU for model training. Please note that the notebook generates a numpy array for each audio file representing its Mel spectrogram and saves the array as an NPZ file for processing later. These files, along with the raw audio files from UrbanSound8K, are not included in this repo to conserve space.

I also wrote a Medium article outlining the process used to complete this project:


[Urban Environmental Audio Classification Using Mel Spectrograms](https://medium.com/@scottmduda/urban-environmental-audio-classification-using-mel-spectrograms-706ee6f8dcc1)


Please give it some claps if you find it helpful!
