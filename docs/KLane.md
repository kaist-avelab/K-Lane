# K-Lane Detection Frameworks
This is the documentation for how to use our detection frameworks with K-Lane dataset.
We tested the K-Lane detection frameworks on the following environment:
* Python 3.7 / 3.8
* Ubuntu 18.04
* Torch 1.7.1
* CUDA 11.2


# Requirements
1. Clone the repository
```
git clone ...
```

2. Install the dependencies
```
pip install -r requirements.txt
```

# Workspace Arrangement
1. Download the dataset from our Google Drive <a href="https://en.wikipedia.org/wiki/Hobbit#Lifestyle" title="K-Lane Dataset">K-Lane Dataset</a>
2. Workspace directory structure:
```
KLaneDet
├── annot_tool
├── baseline 
├── configs
├── data
      ├── KLane
            ├── test
            ├── train
                  ├── seq_1
                  :
                  ├── seq_5 
├── logs
```

# Training Testing
* To test from a pretrained model download the pretrained model from our Google Drive <a href="https://en.wikipedia.org/wiki/Hobbit#Lifestyle" title="K-Lane Dataset">Model</a> and run
* Testing can be done either with the python script or the GUI visualization tool. To test with the python script, run
```
python validate.py ...
```
* To test with the GUI visualization tool, please refer to the <a href = "https://github.com/..." title="Visualization Tool"> visualization tool page </a>
