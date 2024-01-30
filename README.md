# minislam

[![CI](https://github.com/markoelez/minislam/actions/workflows/ci.yaml/badge.svg)](https://github.com/markoelez/minislam/actions/workflows/ci.yaml)


Simple implementation of monocular visual odometry.

![alt text](https://github.com/markoelez/minislam/blob/master/img/example.png?raw=true)
![alt text](https://github.com/markoelez/minislam/blob/master/img/example_two.png?raw=true)


### Usage

run the following for a quick example:
```sh
PYTHONPATH=. python minislam/main.py --dataset=test
```

using your own data:
1. start by doing one of the following:
    - download a sequence of images: https://www.cvlibs.net/datasets/kitti/raw_data.php
    - download an MP4 video
2. move the dataset into the `eval/` directory
3. add dataset metadata to the `config.yaml` file
4. run with `PYTHONPATH=. python minislam/main.py --dataset=[your_dataset_name]`


#### Todo
- add tests
- add graph optimization
- test out other feature detectors
