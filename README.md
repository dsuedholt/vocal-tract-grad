# Vocal Tract Area Estimation by Gradient Descent

Estimating control parameters for the [Pink Trombone](https://dood.al/pinktrombone/) articulatory synthesizer through auto-differentiation.

`resynth.py` is the entry-point script to reproduce a given human recording using the Pink Trombone. 

`physical_tract.py` and `glottis.py` contain simplified synthesis code for the Pink Trombone rewritten in PyTorch.

`tract_proxy.py` calculates the analytical transfer function of a given vocal tract to be used for the loss function for gradient descent.

## Abstract

Articulatory features can provide interpretable and flexible controls for the synthesis of human vocalizations by allowing the user to directly modify parameters like vocal strain or lip position. To make this manipulation through resynthesis possible, we need to estimate the features that result in a desired vocalization directly from audio recordings. In this work, we propose a white-box optimization technique for estimating glottal source parameters and vocal tract shapes from audio recordings of human vowels. The approach is based on inverse filtering and optimizing the frequency response of a wave\-guide model of the vocal tract with gradient descent, propagating error gradients through the mapping of articulatory features to the vocal tract area function. We apply this method to the task of matching the sound of the Pink Trombone, an interactive articulatory synthesizer, to a given vocalization. We find that our method accurately recovers control functions for audio generated by the Pink Trombone itself. We then compare our technique against evolutionary optimization algorithms and a neural network trained to predict control parameters from audio. A subjective evaluation finds that our approach outperforms these black-box optimization baselines on the task of reproducing human vocalizations.