#!/bin/bash


#1. Create a folder for your datasets. Usually, multiple users use one folder for all datasets to be able to share them. Later on, in the 
#training and inference scripts, you will need the path to the dataset.
#2. Create the EML tools folder structure, e.g. ```eml-tools```. The structure can be found here: https://github.com/embedded-machine-learning/eml-tools#interface-folder-structure
ROOTFOLDER=`pwd`

#In your root directory, create the structure. Sample code
mkdir -p projects
mkdir -p venv

#3. Clone the EML tools repository into your workspace
EMLTOOLSFOLDER=./sklearn_ml_toolbox
if [ ! -d "$EMLTOOLSFOLDER" ] ; then
  git clone https://github.com/alexanderwendt/sklearn_ml_toolbox.git "$EMLTOOLSFOLDER"
else 
  echo $EMLTOOLSFOLDER already exists
fi

#4. Create the task spooler script to be able to use the correct task spooler on the device. In our case, just copy
#./init_ts.sh

# Project setup
#5. Create a virtual environment for TF2ODA in your venv folder. The venv folder is put outside of the project folder to 
#avoid copying lots of small files when you copy the project folder. Conda would also be a good alternative.
# From root
cd $ROOTFOLDER

cd ./venv

TF2ODAENV=sklearn
if [ ! -d "$TF2ODAENV" ] ; then
  virtualenv -p python3.8 $TF2ODAENV
  source ./$TF2ODAENV/bin/activate

  # Install necessary libraries
  pip install --no-cache-dir --upgrade pip setuptools cython wheel
  #pip install --no-cache-dir --upgrade setuptools cython wheel
  
  # Install EML libraries
  pip install --no-cache-dir sklearn seaborn pandas pandas-ta numpy matplotlib imbalanced-learn scikit-plot scipy statsmodels umap-learn xgboost missingno backtesting
  
  cd $ROOTFOLDER

  echo Installation complete
  
else 
  echo $TF2ODAENV already exists
fi

cd $ROOTFOLDER
source ./venv/$TF2ODAENV/bin/activate

echo Created TF2ODA environment for TF2ODA inference and OpenVino inference

