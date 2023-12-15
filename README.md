# AeCC: Autoencoders for Compressed Communication

<p align="justify"> In the field of communication systems, the transmission of images over noisy channels poses a significant challenge. To address this challenge, a novel communication system is proposed that employs a vision transformer-based autoencoder for image compression and a denoising network for noise removal. The proposed system operates by first encoding the input image into a lower-dimensional latent space representation using the vision transformer-based autoencoder. This compressed representation is <img align="right" height="150" src="resources/pytorch.png"/> then transmitted through a noisy channel, where it is inevitably corrupted by noise. At the receiver, the denoising network is employed to reconstruct the original image from the received, noisy representation. The denoising network is trained using a dataset of noisy and clean image pairs, enabling it to effectively remove noise artifacts and restore the image's original quality. </p>

## Block Diagram
<img align="center" src="resources/flow.png"/>

## Installation
To get started with this project, follow the steps below:

- Clone the repository to your local machine using the following command:

    ```shell
    git clone https://github.com/muhd-umer/aecc.git
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
    conda activate aecc
    ```

- Install the PyTorch Ecosystem:

    ```shell
    # pip will take care of necessary CUDA packages
    pip3 install torch torchvision torchaudio

    # extra packages
    pip3 install ml_collections einops torchinfo timm
    ```

## Project Structure
The project is structured as follows:

```shell
aecc
├── config/           # configuration directory
├── data/             # data directory
├── model/            # model directory
├── notebooks/        # notebooks directory
├── resources/        # resources directory
├── utils/            # utility directory
├── LICENSE           # license file
├── README.md         # readme file
├── environment.yml   # conda environment file
└── main.py           # main file
```

## Contributing ❤️
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
