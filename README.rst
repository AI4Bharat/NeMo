
Introduction
------------

NVIDIA NeMo Framework is a generative AI framework built for researchers and pytorch developers
working on large language models (LLMs), multimodal models (MM), automatic speech recognition (ASR),
and text-to-speech synthesis (TTS).

This repo implements multi-softmax in ASR models. 

Requirements
------------

1) Python 3.10 or above
2) Pytorch 1.13.1 or above
3) NVIDIA GPU, if you intend to do model training


Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    # create env
    conda create -n temo python=3.10
    conda activate temo

    # clone nemo
    git clone https://github.com/AI4Bharat/NeMo.git
    
    # check the nvcc version and install pytorch
    pip3 install torch torchvision torchaudio
    
    conda install -c nvidia cuda-nvprof=12.1 # Cuda version
    pip install packaging

    # install apex
    git clone https://github.com/NVIDIA/apex
    cd apex
    # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" --config-settings "--build-option=--distributed_adam" --config-settings "--build-option=--deprecated_fused_adam" ./
    
    # install NeMo
    cd ../NeMo
    ./reinstall.sh


If you only want the toolkit without additional conda-based dependencies, you may replace ``reinstall.sh``
with ``pip install -e .`` when your PWD is the root of the NeMo repository.

Mac computers with Apple silicon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To install NeMo on Mac with Apple M-Series GPU:

- create a new Conda environment

- install PyTorch 2.0 or higher

- run the following code:

.. code-block:: shell

    # [optional] install mecab using Homebrew, to use sacrebleu for NLP collection
    # you can install Homebrew here: https://brew.sh
    brew install mecab

    # [optional] install pynini using Conda, to use text normalization
    conda install -c conda-forge pynini

    # install Cython manually
    pip install cython

    # clone the repo and install in development mode
    git clone https://github.com/AI4Bharat/NeMo.git
    cd NeMo
    pip install 'nemo_toolkit[all]'

    # Note that only the ASR toolkit is guaranteed to work on MacBook - so for MacBook use pip install 'nemo_toolkit[asr]'

Windows Computers
~~~~~~~~~~~~~~~~~

One of the options is using Windows Subsystem for Linux (WSL).

To install WSL:

- In PowerShell, run the following code:

.. code-block:: shell

    wsl --install
    # [note] If you run wsl --install and see the WSL help text, it means WSL is already installed.

Learn more about installing WSL at `Microsoft's official documentation <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

After Installing your Linux distribution with WSL:
  - **Option 1:** Open the distribution (Ubuntu by default) from the Start menu and follow the instructions.
  - **Option 2:** Launch the Terminal application. Download it from `Microsoft's Windows Terminal page <https://learn.microsoft.com/en-us/windows/terminal>`_ if not installed.

Next, follow the instructions for Linux systems, as provided above. For example:

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

RNNT
~~~~
Note that RNNT requires numba to be installed from conda.

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c conda-forge numba

Apex
~~~~
To install Apex, please follow the following URL: https://github.com/NVIDIA/apex.git

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Apex or any other dependencies.

While installing Apex, it may raise an error if the CUDA version on your system does not match the CUDA version torch was compiled with.
This raise can be avoided by commenting it here: https://github.com/NVIDIA/apex/blob/master/setup.py#L32

cuda-nvprof is needed to install Apex. The version should match the CUDA version that you are using:

.. code-block:: bash

  conda install -c nvidia cuda-nvprof=11.8

packaging is also needed:

.. code-block:: bash

  pip install packaging

With the latest versions of Apex, the `pyproject.toml` file in Apex may need to be deleted in order to install locally.

Examples
--------

Many examples can be found under the `"Examples" <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_ folder.
