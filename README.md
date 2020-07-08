# DCGRU_Tensorflow2
Alternative implementation of the DCGRU recurrent cell in Tensorflow 2

Diffusion Convolutional Gated Recurrent Unit (DCGRU) is the recurrent cell adopted in the Diffusion Convolutional Recurrent Neural Network (DCRNN), a deep learning architecture introduced by Y. Li et al. [1] to tackle spatiotemporal forecasting tasks, where data is represented as a graph signal.
The DCGRU cell is a modification of the standard GRU, having each matrix multiplication operation replaced by the the diffusion convolution operation (see [1]).

(Since I'm not part of the authors of the original DCRNN paper [1], this should be treated as an unofficial implementation)

# How to build a DCGRU layer

Define the dcgru cell:
```python
from tensorflow import keras
from dcgru_cell_tf2 import DCGRUCell

dcgru_cell = DCGRUCell(units=20,adj_mx=G_adj_mx, K_diffusion=2,
                       num_nodes=num_nodes,filter_type="random_walk")
```

Wrap the dcgru cell in a keras RNN layer:
```python
dcgru_layer = keras.layers.RNN(dcgru_cell)
```


# DCGRU basic example

A Jupyter notebook showing a basic usage of a DCGRU layer (DCGRU_example.ipynb)

https://colab.research.google.com/drive/1sq0s8j3gGpBvSq058aBcrvSX2wLuAE-t?usp=sharing


# DCGRU testing

A Jupyter notebook with comparison of the DCGRU with other common neural networks (DCGRU_testing.ipynb)

https://colab.research.google.com/drive/1-Em1U6jJZO7yOL8Uimgm5o0vkLIPP0dE?usp=sharing



# References
[1] Li, Y., Yu, R., Shahabi, C., Liu, Y.: Diffusion convolutional recurrent neural net-work: Data-driven traffic forecasting. arXiv preprint arXiv:1707.01926 (2017)

https://github.com/liyaguang/DCRNN
