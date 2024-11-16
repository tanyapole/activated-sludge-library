# Activated Sludge Library

### About
This repository contains Activated Sludge Library - an open-source library for assisting in analysis of activated sludge microorganisms. 

The main feature implemented in the library is the analysis of microscopy images of activated sludge, including
1. recognition of presence of the following microorganisms: Annelida, Ciliophora, Nematoda, Rotifera, Sarcodina;
2. localization of present microorganisms (with bounding boxes).

### How to install

### How to use
```
from PIL import Image
import sludge

analyzer = sludge.SludgeAnalyzer()
img = Image.open(<path_to_image>)
res = analyzer.predict(img) # result
```
