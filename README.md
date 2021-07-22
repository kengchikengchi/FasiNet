# FasiNet
Real-Time Neural Image Compression in a Non-GPU Environment

ACM Reference Format:
Zekun Zheng, Xiaodong Wang, Xinye Lin, and Shaohe Lv. 2021.
Get the Best of the Three Worlds: Real-Time Neural Image Compression in a Non-GPU Environment.
In Proceedings of the 29th ACM International Conference on Multimedia (MM '21),
October 20-s24, 2021, Virtual Event, China.
https://doi.org/10.1145/3474085.3475667.

## Paper Summary
Lossy image compression always faces a tradeoff between rate-distortion performance and compression/decompression speed. With the advent of neural image compression, hardware (GPU) becomes the new vertex in the tradeoff triangle. By resolving the high GPU dependency and improving the low speed of neural models, this paper proposes two non-GPU models that get the best of the three worlds. First, the CPU-friendly Independent Separable Down-Sampling (ISD) and Up-Sampling (ISU) modules are proposed to lighten the network while ensuring a large receptive field. Second, an asymmetric autoencoder architecture is adopted to boost the decoding speed. At last, the Inverse Quantization Residual (IQR) module is proposed to reduce the error caused by quantization. In terms of rate-distortion performance, our network surpasses the state-of-the-art real-time GPU neural compression work at medium and high bit rates. In terms of speed, our model's compression and decompression speeds surpass all other traditional compression methods except JPEG, using only CPUs. In terms of hardware, the proposed models are CPU friendly and perform stably well in a non-GPU environment.

### Environment 

* Python==3.6
* Tensorflow==1.14.0
* [Tensorflow-Compression](https://github.com/tensorflow/compression) ==1.2
```
    pip3 install tensorflow-compression==1.2 or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
```

### Usage

For training, set step to 0.

For compressing, set step to 1.

For decompressing set step to 2.

We recommend that you use GPU for training and CPU for testing
