# Wavelet-Final
repo for wavelets and filter banks final project

There's a lot of applications for speaker recognition, but not many people have mastered this art. We decide to give it a try with what we've learned so far from wavelets and filter banks, MLSP, and CV. We hope to generate an algorithm that is capable of detecting the identity of the speaker from a small audio signal segmentation. We will do the following to approach this idea.  

1. Literature review (the goal is to find the best work done before)  
  a. some spec:  
    - good accuracy
    - provide all parameters (DWT design, NN model parameters such as error function)
    - provide datasets
  b. find the following take-away from literatures  
    - what are the state-of-the-art methods to do speaker recognition
    - how to construct a NN (maybe scattering transform) to learn good FBs
    - how to construct a NN (maybe LSTM) to learn good wavelet coeff for classification
    - how to design good FBs
    - good DWT structure for speaker recognition

2. Datasets search  
  a. from literatures  
  b. from google  
  
3. Develop a working model (feature extraction, training, and classification)  
  a. use a fixed DWT structure and FBs  
  b. use a DWT coeff as inputs to a NN model (from literature)  
  c. classify and calculate accuracy  
  
4. Fine tune  
  a. Finalize NN model that learn good wavelet coeff  
  b. Try different structures, finalize structure  
  c. Finalize NN model that learn good FB  

5. Comparison
  by now, we should have an optimized working model (feature extraction, training, and classification)  
  a. fix training and classification methods, try different feature extraction methods  
    - MFCC
    - STFT
    - Wavelet packets
  
