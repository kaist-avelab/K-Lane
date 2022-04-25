![image](./docs/imgs/logo_high_resolution.png)


`K-Lane` is the world's first open LiDAR lane detection frameworks that provides a dataset with wide range of driving scenarios in an urban environment. This repository provides the `K-Lane` frameworks, annotation tool for lane labelling, and the visualization tool for showing the inference results and calibrating the sensors.


![image](./docs/imgs/klane_examples.png)


# K-Lane Detection Frameworks
This is the documentation for how to use our detection frameworks with K-Lane dataset.
We tested the K-Lane detection frameworks on the following environment:
* Python 3.7 / 3.8
* Ubuntu 18.04
* Torch 1.7.1
* CUDA 11.2

# Preparing the Dataset
1. To download the dataset, log in to <a href="https://kaistavelab.direct.quickconnect.to:54568/"> our server </a> with the following credentials: 
      ID       : klaneds
      Password : Klane2022
2. Go to the "File Station" folder, and download the dataset by right-click --> download.
   Note for Ubuntu user, there might be some error when unzipping the files. Please check the "readme_to_unzip_file_in_linux_system.txt".
3. After all files are downloaded, please arrange the workspace directory with the following structure:
```
KLaneTools
├── annot_tool
├── baseline 
├── configs
      ├── config_vis.py
      ├── Proj28_GFC-T3_RowRef_82_73.py
      ├── Proj28_GFC-T3_RowRef_82_73.pth
├── data
      ├── KLane
            ├── test
            ├── train
                  ├── seq_1
                  :
                  ├── seq_15 
├── logs
```
![image](./docs/imgs/download_manual.png)

---
# The total development kits for K-Lane (training, evaluation code, annotation and visualization tools) are published soon via this repository! (until the end of April)

## Requirements

1. Clone the repository
```
git clone ...
```

2. Install the dependencies (TBD)
```
pip install -r requirements.txt
```

## Training Testing
* To test from a pretrained model (e.g., Proj28_GFC-T3_RowRef_82_73.pth), download the pretrained model from our Google Drive <a href="https://drive.google.com/drive/folders/14QHSxbCsUEf0FYZIa3j_uMFcLmMDwQmB?usp=sharing" title="K-Lane Dataset">Model</a> and run
```
python validate_gpu_0.py ...
```
* Testing can be done either with the python script or the GUI visualization tool. To test with the GUI visualization tool, please refer to the <a href = "https://github.com/..." title="Visualization Tool"> visualization tool page </a>
*  Youtube Link (TBD)

## Development Kit 
1. [Visualization Tool](./docs/visualization.md)
      * [Requirements](./docs/visualization.md#Requirements)
      * [Workspace Arrangement](./docs/visualization.md#Workspace-Arrangement)
      * [Visualizing Inference](./docs/visualization.md#Visualizing-Inference)
      * [Calibrating LiDAR to Camera](./docs/visualization.md#Calibrating-LiDAR-to-Camera)
2. [Annotation Tool](./docs/annotation.md)
      * [Requirements](./docs/annotation.md#Requirements)
      * [Workspace Arrangement](./docs/annotation.md#Workspace-Arrangement)
      * [Labelling a Point Cloud](./docs/annotation.md#Labelling-a-Point-Cloud)

## Updates
* [2022-04-18] v1.0.0 is released along with the K-Lane Dataset. Please check [Getting Started](./docs/KLane.md#Workspace-Arrangement) for the download instruction.

## License
`K-Lane` is released under the Apache-2.0 license.

## Acknowledgement
The K-lane benchmark is contributed by `Dong-Hee Paek`, `Kevin Tirta Wijaya`, `Dong-In Kim`, and `Min-Hyeok Sun`, advised by `Seung-Hyun Kong`.

We thank the maintainers of the following projects that enable us to develop `K-Lane`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`TuRoad`](https://github.com/Turoad/lanedet) bu TuZheng

## Model Zoo
(TBD)

## Citation

If you find this work is useful for your research, please consider citing:
```
@InProceedings{paek2022klane,
  title     = {K-Lane: Lidar Lane Dataset and Benchmark for Urban Roads and Highways},
  author    = {Paek, Dong-Hee and Kong, Seung-Hyun and Wijaya, Kevin Tirta},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022}
}
```
