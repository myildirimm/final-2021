# Environment Setup

Download Simstar from the link:

https://drive.google.com/open?id=1Gp-XXnOX9dbDcfqFJNJ4UtZqo9sWqjUg



Checkout [environment.md](environment.md) for details. 

### Windows 
Just click on Simstar.exe and Simstar is ready.


### Linux 

## Requirements
Install Vulkan Libraries
	
	sudo apt-get install vulkan-utils

    cd Simstar
  
    chmod 777 -R *
  
    ./Simstar.sh



## Requirements

### Python Package Requirements

#### Option 1: Install using Anaconda
Create a new environment using anaconda. 

	conda env create --file environment.yml

	conda activate simstar-env


#### Option 2: Install using pip
	
Install required python libraries from requirements.txt by

	pip install -r requirements.txt


### Pytorch Version

The final evaluation will be using pytorch version 1.5 and CUDA version 10.2.


## Install Python API

      cd PythonAPI

      python setup.py install


## Installation Test

There are multiple stages that needs to be checked. 

### 1. Test Simstar Executable

Open the simstar executable, allow for networking if asked. 

![opening_screen](PythonAPI/img/opening.png)

### 2. Test PythonAPI installation

Run the following with success.

	cd PythonAPI

	python python_api_intro.py

### 3. Test Environment Setup

	cd sac_example

	python example_experiment.py


### Optional Test

To test a closed loop training with Pytorch, you can run the example SAC agent from examples folder.

	cd sac_example

	python train_example.py

