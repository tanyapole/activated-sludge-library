# Activated Sludge Library

### About
This repository contains Activated Sludge Library - an open-source library for assisting in analysis of activated sludge microorganisms. 

The main feature implemented in the library is the analysis of microscopy images of activated sludge, including
1. recognition of presence of the following microorganisms: Annelida, Ciliophora, Nematoda, Rotifera, Sarcodina;
2. localization of present microorganisms (with bounding boxes).

### How to install
It is recommended to perform installation inside an isolated environment. <br/>
E.g. you can execute following commands to create and activate a conda environment:
```
conda create -n sludge python=3.10
conda activate sludge
```
Install necessary libraries:
```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.3.27
```
Optionally you can create a jupyter kernel for the environment:
```
pip install ipython ipykernel
python -m IPython kernel install --user --name sludge
```
### How to use
```
from PIL import Image
import sludge

analyzer = sludge.SludgeAnalyzer()
img = Image.open(<path_to_image>)
res = analyzer.predict(img) # result
```
