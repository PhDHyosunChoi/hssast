# [HSSAST Dual representations: A novel variant of Self-Supervised Audio Spectrogram Transformer with multi-layer feature fusion and pooling combinations for sound classification](https://www.sciencedirect.com/science/article/pii/S0925231225000876)

### Authors: Hyosun Choi, Li Zhang, Chris Watkins

## Highlights :
### • We propose an SSAST variant with multilayer feature fusion and pooling combination. 
### • A variety of all-patch-wise min, max, and mean pooling permutations are applied.
### • Dual representations based on multi-layer feature fusion are produced.
### • Our model depicts great superiority for tackling audio classification tasks.


## Abstract :
The Self-Supervised Audio Spectrogram Transformer (SSAST) has recently been verified as the state-of-the-art model in various audio and speech command classification tasks. SSAST uses self-supervised learning to reduce the need of substantial data to pre-train transformers, removing the disadvantage of its supervised learning counterpart, the Audio Spectrogram Transformer (AST) model. Owing to the fact that transformers such as SSAST use only feature representations from the last layer for downstream classification tasks, we believe that such a process will lose some important information from middle layers during training. Therefore, in this research, we propose a novel variant of the SSAST model using a dual representation generated using fusion of the outputs from multi-layers (i.e. both middle and last layers) for audio classification. Specifically, we apply all-patch-wise pooling combinations to all patches from both a middle layer and the last layer of a pre-trained patch-based self-supervised learning model. As such, it generates two individual sequences of the output patches based on a variety of mean, max, and min pooling combinations to make the final double-sized representation. This dual representation includes more discriminative information and better knowledge, providing the linear multi-layer perceptron head layers with more useful information for audio classification. In comparison with existing state-of-the-art studies, the proposed model using the dual representations yielded by multi-layer feature fusion and pooling combinations significantly boosts performance on all tasks. The resulting accuracy rates are 93.67%, 100%, 79.59%, 79.59%, 91.22%, and 85.90% for CREMA-D, TESS, RAVDESS, Speech Emotion Classification, Isolated Urban Events, and CornellBirdCall, respectively.

### Published Paper at Neurocomputing Volume 623, 28 March 2025, 129415 : https://www.sciencedirect.com/science/article/pii/S0925231225000876
