# Flower App Environment and Execution Instructions 
### Create Flower APP environment
**Step 1. Download Anaconda on machine:**  
Click here for download:  
https://www.anaconda.com/products/individual

**Step 2. Open Anaconda Terminal Window:**  
Once Anaconda is installed, depending on OS, please open the Anaconda terminal Window
*Windows Machine* ---> Located in Anaconda Navigator.   
*Mac OS* ---> Mac Terminal Application. 

**Step 2. Create Anaconda Enviroment for FLower Application:**  
Update conda to create a new environment. When creating the enviroment please ensure  
that python version is below 3.9 for tensorflow to be executed without error. For this instruction we advise python version 3.7
```sh
conda update
conda create -n flowerEnv python=3.7 anaconda
```
**Step 2. Activate the conda enviroment:**  
In this step, we will now activate out new Conda environment to install python packages. 

> Note: `In the furture if you need to install more packages please run this command so that you ensure the packages are being install on the conda enviroment you prefer`

```sh
conda activate FlowerEnv
```
**Step 3. Install Pillow Package:**  
Install Python Image library on FlowerEnv enviroment
*https://python-pillow.org/*
```sh
conda install pillow
```
**Step 4. Install Tensorflow:**  
Install Tensoflow Packages
```sh
conda install tensorflow
```
or
```sh
conda install tensorflow
```
**Step 5. Install Keras:**  
Install Keras Packages
```sh
conda install -c anaconda keras
```
or
```sh
conda install keras
```
**Step 6. Install Tkinter:**     
Install Tkinter GUI
```sh
conda install -c anaconda tk
```
or
```sh
conda install tk
```
### Download PyCharm Community IDE
***Community version is free*** 
https://www.jetbrains.com/pycharm/download  

### Clone Github Project 
**Step 1. Git Version Control Installation**  
*Git Version control installation step *
https://github.com/git-guides/install-git

**Step 2. Clone FlowerApp Repo**
```sh
git clone https://github.com/pkmohabir1/flowerAPP.git
```

### Execute Flower Application
**Step 1. Git Pull directories and files**.  
Open PyCharm IDE  

**Step 2. Add your Conda Enviroment to the PyCharm interpreter**  
*Follow Instructions here*  
https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html 

**Step 3. Add your Conda Enviroment to the PyCharm interpreter**  
Now open the local directory folder (Cloned github Repo) on Pycharm  

**Step 4. Add your Conda Enviroment to the PyCharm interpreter**  
Ensure that your python interpreter/virtual enviroment is set to the Conda FlowerEnv we     created in the ***Create Flower APP environment*** section.  

**Step 5. Run Python Flower Python script**  
Simply click on PyCharm run button, and execute the ***Flower Python*** script file. 


