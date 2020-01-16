---
layout: post
title: Tensorflow2-GPU Windows installation guide
subtitle: Build your deep learning development environment
tags: [deeplearning]
comments: false
mathjax: false
---

1. OS and GPU check

   Go to [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus) and verify compute capability of your GPU meet the minimal requirement (compute power 3.0 or higher).

2. Update NVIDIA graphical card driver

   You may need to update the GPU driver either from Windows device manager or Nvidia control panel.

3. Install CUDA toolkit

   Download NVIDIA CUDA Toolkit from [CUDA downloads site](http://developer.nvidia.com/cuda-downloads) and install the executable as prompted. Below used latest version v10.2 as an example.

4. Install CUDNN

   Download NVIDIA cuDNN from [this](https://developer.nvidia.com/cudnn) site, you are required to register for the [NVIDIA Developer Program](https://developer.nvidia.com/accelerated-computing-developer) before downloading the resources. Then unzip the cudnn package and place `cudnn64_7.*.dll`, `cudnn.h` and `cudnn.lib` to CUDA installation path, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include` and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64`, respectively.

5. Check system environment path

   Ensure `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2` is set and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\CUPTI\libx64`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include` are in system `Path`.

6. Install MS Visual Studio compiler

   Download and install [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/), use community version and tick desktop development component to install when prompted may be sufficient in most cases. To test above installations, open a CUDA sample solution or project (default location `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2`), right-click on the project name, select property, go to Linker and add `cudnn.lib`, then build the solution/project. If no error occurred, you may run the generated executable whose location can be found from the build log. Now you are ok to preceed.

7. Install python and set up virtualenv

   Visit [Python.org](https://python.org) to download and install latest python 3 release. Tick `adding to system path` if you want Windows system automatically find python.exe every time you launch python. After installation, you can open a command prompt or windows powershell, type `python` and hit enter.

   Python is powered by a rich collection of modules. To increase connection speed when use `pip install` to install modules, type and execute to change default module mirror:

   ```sh
   python -m pip install -U pip
   pip config set global.site-url "https://pypi.tuna.tsinghua.edu.cn/simple"
   ```

8. Install tensorflow-gpu

   It's more practical and safer to use virtual environments to organize development environments:

   ```sh
   pip install -U virtualenv
   ```

   then

   ```sh
   python -m virtualenv tf-gpu
   .\Scripts\activate
   pip install tensorflow-gpu
   ```

   Finally check it's working properly by:

   ```sh
   python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
   ```

   It will return `True` if everything works fine.
