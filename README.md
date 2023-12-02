[<img align="right" width="235" height="270" src="resources/nust.svg"/>](https://nust.edu.pk/)

# ViTCC: Vision Transformer Enabled Communication Compression

<p align="justify"> In the field of communication systems, the transmission of images over noisy channels poses a significant challenge. To address this challenge, a novel communication system is proposed that employs a vision transformer-based autoencoder for image compression and a denoising network for noise removal. The proposed system operates by first encoding the input image into a lower-dimensional latent space representation using the vision transformer-based autoencoder. This compressed representation is <img align="right" width="235" height="125" src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png"/> then transmitted through a noisy channel, where it is inevitably corrupted by noise. At the receiver, the denoising network is employed to reconstruct the original image from the received, noisy representation. The denoising network is trained using a dataset of noisy and clean image pairs, enabling it to effectively remove noise artifacts and restore the image's original quality. </p>

## Block Diagram
<img align="center" src="resources/flow.png"/>

## Installation
To get started with this project, follow the steps below:

- Clone the repository to your local machine using the following command:

    ```shell
    git clone https://github.com/muhd-umer/vtcc.git
    ```

- It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects. To create a new virtual environment, run the following command:

    ```shell
    conda env create -f environment.yml
    ```

- Alternatively, you can use `mamba` (faster than conda) package manager to create a new virtual environment:

    ```shell
    conda install mamba -n base -c conda-forge
    mamba env create -f environment.yml
    ```

- Activate the newly created environment:

    ```shell
    conda activate vtcc
    ```

- Install JAX+Flax

    ```shell
    pip install --upgrade pip

    # CUDA 12 installation
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # Other packages
    pip install flax ml_collections clu termcolor
    ```

## License
```
MIT License

Copyright (c) 2023 Muhammad Umer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
