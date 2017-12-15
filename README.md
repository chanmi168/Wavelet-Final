# Wavelet-Final
repo for wavelets and filter banks final project

There's a lot of applications for speaker recognition, but not many people have mastered this art. We decide to give it a try with what we've learned so far from wavelets and filter banks, MLSP, and CV. We hope to generate an algorithm that is capable of detecting the identity of the speaker from a small audio signal segmentation. We will do the following to approach this idea.  

1. Literature review (the goal is to find the best work done before)  
  * some spec:  
    - good accuracy
    - provide all parameters (DWT design, NN model parameters such as error function)
    - provide datasets
    
    
2. find the following take-away from literatures
 * what are the state-of-the-art methods to do speaker recognition
   - MFCC feature extractions, HMM/GMM/ANN/ICSI system/NIST
 * how to construct a NN (maybe scattering transform) to learn good FBs
   - not found
 * how to construct a NN (maybe LSTM) to learn good wavelet coeff for classification
   - LSTM/bidirectional LSTM/GRU/GRNN/
 * how to design good FBs
   - MFCC like, Doubenchies filters (64 order, 7 level)
 * good DWT structure for speaker recognition
   - wavelet packet, irregular decomposition, simply DWT

3. Datasets search  
  * from literatures  
  * from google  
  
4. Develop a working model (feature extraction, training, and classification)  
  * use a fixed DWT structure and FBs  
  * use a DWT coeff as inputs to a NN model (from literature)  
  * classify and calculate accuracy  
  
5. Fine tune (the order is important, please fix one first)  
  * Finalize NN model that learn good wavelet coeff  
  * Try different structures, finalize structure  
  * Finalize NN model that learn good FB  

6. Comparison
  by now, we should have an optimized working model (feature extraction, training, and classification)  
  * fix training and classification methods, try different feature extraction methods  
    - MFCC
    - STFT
    - Wavelet packets
  
