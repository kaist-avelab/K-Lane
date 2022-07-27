<p align="center">
  <img src = "./docs/imgs/klane_logo_ver2.png" width="60%">
</p>

`K-Lane (KAIST-Lane)` (provided by [`AVELab`](http://ave.kaist.ac.kr/)) is the world's first open LiDAR lane detection frameworks that provides a dataset with wide range of driving scenarios in an urban environment. This repository provides the `K-Lane` frameworks, annotation tool for lane labelling, and the visualization tool for showing the inference results and calibrating the sensors.


![image](./docs/imgs/klane_examples.png)


# K-Lane Detection Frameworks
This is the documentation for how to use our detection frameworks with K-Lane dataset.
We tested the K-Lane detection frameworks on the following environment:
* Python 3.7 / 3.8
* Ubuntu 18.04
* Torch 1.7.1
* CUDA 11.2

## Preparing the Dataset
You can get a dataset in two different ways. One is about how to use our server, and the other is about how to use Google Drive.

* Via our server

1. To download the dataset, log in to <a href="https://kaistavelab.direct.quickconnect.to:54568/"> our server </a> with the following credentials: 
      ID       : klaneds
      Password : Klane2022
2. Go to the "File Station" folder, and download the dataset by right-click --> download.
   Note for Ubuntu user, there might be some error when unzipping the files. Please check the "readme_to_unzip_file_in_linux_system.txt".
3. After all files are downloaded, please arrange the workspace directory with the following structure:
```
KLaneFrameworks
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
            ├── description_frames_test.txt
            ├── description_test_lightcurve.txt
├── logs
```
![image](./docs/imgs/download_manual.png)

* Via Google Drive Urls

Also, you can get the dataset through the following Google Drive urls.

1. <a href="https://drive.google.com/drive/folders/1NE9DT8wZSbkL95Z9bQWm22EjGRb9SsYM?usp=sharing" title="K-Lane Dataset">link for download seq_01 to 04</a> 
2. <a href="https://drive.google.com/drive/folders/1YBz5iaDLAcTH5IOjpaMrLt2iFu2m_Ui_?usp=sharing" title="K-Lane Dataset">link for download seq_05 to 12</a>
3. <a href="https://drive.google.com/drive/folders/1dUIYuOhnKwM1Uf5Q-nC0X0piCZFL8zCQ?usp=sharing" title="K-Lane Dataset">link for download seq_13 to 14</a>
4. <a href="https://drive.google.com/drive/folders/12aLITzR_tE8eVi-Q4OWoomX9VR3Olea7?usp=sharing" title="K-Lane Dataset">link for download seq_15, test, and description</a>

## Requirements

1. Clone the repository
```
git clone ...
```

2. Install the dependencies
```
pip install -r requirements.txt
```

## Training & Testing
* To train the model, prepare the total dataset and run
```
python train_gpu_0.py ...
```
* To test from a pretrained model (e.g., Proj28_GFC-T3_RowRef_82_73.pth), download the pretrained model from our Google Drive <a href="https://drive.google.com/drive/folders/14QHSxbCsUEf0FYZIa3j_uMFcLmMDwQmB?usp=sharing" title="K-Lane Dataset">Model</a> and run
```
python validate_gpu_0.py ...
```
* Testing can be done either with the python script or the GUI visualization tool. To test with the GUI visualization tool, please refer to the <a href = "https://github.com/..." title="Visualization Tool"> visualization tool page </a>
*  Youtube Movie for 'How to use the annotation tool': https://www.youtube.com/watch?v=FQgXLigdgxY&t=12s

## Model Zoo
|Name|Overall|Daylight|Night|Urban|Highway|Curve|Merging|Occ-0|Occ-2|Occ-4~6|GFLOPs|Model|Paper|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|LLDN-GFC|82.12|82.22|82.00|81.75|82.55|78.05|81.08|82.97|81.28|75.92|558.0|<a href="https://drive.google.com/drive/folders/14QHSxbCsUEf0FYZIa3j_uMFcLmMDwQmB?usp=sharing">Link</a>|<a href="https://openaccess.thecvf.com/content/CVPR2022W/WAD/papers/Paek_K-Lane_Lidar_Lane_Dataset_and_Benchmark_for_Urban_Roads_and_CVPRW_2022_paper.pdf">Link</a>|
|RLLDN-LC|82.74|82.58|82.92|81.64|84.05|76.16|79.92|83.44|82.00|79.16|387.5|<a href="https://drive.google.com/drive/folders/14QHSxbCsUEf0FYZIa3j_uMFcLmMDwQmB?usp=sharing">Link</a>|Link|

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
The K-lane benchmark is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=7), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

We thank the maintainers of the following projects that enable us to develop `K-Lane`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`TuRoad`](https://github.com/Turoad/lanedet) bu TuZheng.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2021R1A2C3008370).

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
