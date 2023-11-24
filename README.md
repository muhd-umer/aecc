# VTCC


## Installation
To get started with this project, follow the steps below:

**Clone the repository**
- Clone the repository to your local machine using the following command:
```shell
git clone https://github.com/muhd-umer/vtcc.git
```

**Create a new virtual environment**
- It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects. To create a new virtual environment, run the following command:
```shell
conda env create -f environment.yml
```

- Alternatively, you can use `mamba` (faster than conda) package manager to create a new virtual environment:
```shell
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```

**Install the dependencies**
- Activate the newly created environment:
```shell
conda activate vtcc
```

- Install PyTorch (Stable 2.0.1):
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
