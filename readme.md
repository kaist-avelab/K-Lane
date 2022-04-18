![image](./docs/imgs/logo_high_resolution.png)


`K-Lane` is the world's first open LiDAR lane detection frameworks that provides a dataset with wide range of driving scenarios in an urban environment. This repository provides the `K-Lane` frameworks, annotation tool for lane labelling, and the visualization tool for showing the inference results and calibrating the sensors.

## Overview 
1. [K-Lane Detection Frameworks](./docs/KLane.md)
      * [Requirements](./docs/KLane.md#Requirements)
      * [Workspace Arrangement](./docs/KLane.md#Workspace-Arrangement)
      * [Training and Testing with K-Lane](./docs/KLane.md#Training-Testing)
2. [Visualization Tool](./docs/visualization.md)
      * [Requirements](./docs/visualization.md#Requirements)
      * [Workspace Arrangement](./docs/visualization.md#Workspace-Arrangement)
      * [Visualizing Inference](./docs/visualization.md#Visualizing-Inference)
      * [Calibrating LiDAR to Camera](./docs/visualization.md#Calibrating-LiDAR-to-Camera)
3. [Annotation Tool](./docs/annotation.md)
      * [Requirements](./docs/annotation.md#Requirements)
      * [Workspace Arrangement](./docs/annotation.md#Workspace-Arrangement)
      * [Labelling a Point Cloud](./docs/annotation.md#Labelling-a-Point-Cloud)

## Updates
* [2021-09-11] v1.0.0 is released

## License
`K-Lane` is released under the MIT License

## Acknowledgement
The dataset is made by `Donghee Paek`, `Kevin Tirta Wijaya`, `Dongin Kim`, and `Minhyeok Sun`, supervised by `Seunghyun Kong`

We thank the maintainers of the following projects that enable us to develop `K-Lane`:
* [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB

## Citation

If you find this work is useful for your research, please consider citing:
```
@article{paek2021mixer,
  title={Mixer-based lidar lane detection network and dataset for urban roads},
  author={Paek, Donghee and Kong, Seung-Hyun and Wijaya, Kevin Tirta},
  journal={arXiv preprint arXiv:2110.11048},
  year={2021}
}
```
