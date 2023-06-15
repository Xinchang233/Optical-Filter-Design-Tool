# An App for Filter Design & Optimization
### Introduction
This app helps you extract the optimal design of filter, as illustrated in the paper (i.e. minimum insertion loss, optimal transfer function shape, etc.). It is constructed using [dash](https://dash.plotly.com), a Python library that supports interactive graphing. Most importantly, this app is farely easy to use. Though it is published as source codes, only few steps needs to be done for setup, which will be illustrated in the following section.
### Environment setup & Installation
1. To make the code running, Python along with some affiliated packages needs to be installed on your own computer. This can be done easily with the help of [Anaconda](https://www.anaconda.com). To install it, follow the link and go to its official website and download the version corresponding to your operational system. It is same as installing any other software on your own machine.  
2. After sucessfully installing Anaconda, the environment should be set up properly automatically. Since the packages used in the code is usually preinstalled along with Anaconda.  
3. Then download the code and put it under some convinient directory on your computer.  
4. Open the Terminal of your computer, type `conda activate` and Enter. If you are using Windows, then open Anaconda Prompt and do the same thing.  
5. Type `python _directory_of_your_downloaded_code_/filter_visualize.py` then Enter.  
6. Several new lines will appear on the screen. Copy the URL (in the red box of figure below) and paste it into browser. The app is available to you. ![illustration1](https://github.com/Xinchang233/Dual-Ring-Filter-Calculator/blob/main/illustration1.png)  
##### Trouble shooting
If some error like `ModuleNotFoundError: No module named 'dash'` appears on screen, following [this](https://dash.plotly.com/installation) to install dash.
### Demostration
