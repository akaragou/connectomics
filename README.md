# Connectomics

2D and 3D segmentation of neural processes in EM images. 

---
![plot](https://user-images.githubusercontent.com/16754088/38761707-13cd0944-3f53-11e8-8e0a-4944d0177eea.png)
---
3D segementations coming soon
---
- class\_weights\_Berson.npy - class weights for balancing classes
- config.py - stores values for filepaths and model parameters
- prepare\_tf\_records\_Berson.py - creates tf records for Berson data
- prepare\_tf\_records\_ISBI\_2012.py - creates tf records for ISBI 2012 challenge data
- test.py - evaluates model on test data plotting resulting masks
- tf\_record.py - Contains methods to create and decode tf records
- tiramisu.py - 100 layer  Fully Convolutional Densnet Implementation
- train.py - trains segmentation model from scratch
- unet.py - implementation of modified unet with dropout in the upsampling path
