# AI/ML project for superconductor analysis

Please refer to LICENSE 
Author: adam.ray.zheng@gmail.com & Xie, Weiwei <xieweiwe@msu.edu>

## Data preparation

The superconducting data for alloys were retrieved from the database of superconducting materials (SUPERCON) https://github.com/vstanev1/Supercon for simulation and comparison. For each data point, a data vector of length 120 is generated with a position index representing the elements and its corresponding normalized percentage (such that the sum of all percentages of all elements in the compound is 1) being filled into the vector. The dataset consists of 20000 entries, which are used for training, validation, and testing. Consequently, a 2D data array of dimensions 20000 × 120 is created. The ground truth comprises two components. The first component is the classification ground truth, represented by a single binary value of 0 or 1, indicating the presence or absence of superconductivity. Thus, the two classes correspond to non-superconducting compounds (0) and superconducting compounds (1). The second component is the measured Tc value.


## Model Construction. 

The DNN model comprises two components: the network backbone and prediction. Prediction involves two branches: one for classification and the other for Tc value prediction.

The classification branch is responsible for determining whether a compound exhibits superconductivity. This branch employs three layers: a 3×3 convolution layer, a GEMM layer, and a SoftMax layer. The output of the SoftMax calculation is a 1×2 tensor, where the first value (Prob0) represents the probability of the compound being non-superconducting, and the second value (Prob1) represents the probability of the compound being superconducting.33 The classification result is determined by comparing Prob0 and Prob1. If Prob1 > Prob0, the compound is classified as superconductive. Otherwise, it is classified as non-superconductive. It is noteworthy that the sum of Prob0 and Prob1 is equal to 1.

The Tc prediction branch is responsible for predicting the Tc value of a compound. This branch consists of multiple convolution layers, each followed by a ReLU layer: two 3×3 convolution layers and two 1×1 convolution layers, followed by two GEMM layers. The output of this branch is a 1×1 tensor that indicates the predicted Tc value in Kelvin. The convolution layers are a fundamental building block of convolution networks, and they operate by exploiting the local connectivity of multi-dimensional input data to generate the output. The convolution kernel, also referred to as the receptive field, defines the spatial extent of this local connectivity. The convolution layer acts as an information fusion and abstraction block that enables the extraction of meaningful features from the input data.


## Traing and Testing 

The training process is carried out in two stages. First, the Backbone and Tc Prediction branches are jointly trained, and the loss of the predicted Tc value versus the ground truth is used for network backpropagation. Once the Tc Prediction results reach a satisfactory level, the parameters of the Backbone and Tc Prediction branches are fixed. The training is then performed solely on the Classification branch.

## Result Analysis and Model Complexity

The testing process involves evaluating the performance of the trained model on a separate dataset known as the test dataset, which is distinct from the data used during model training


