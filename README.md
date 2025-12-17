# minislam

[![CI](https://github.com/markoelez/minislam/actions/workflows/ci.yml/badge.svg)](https://github.com/markoelez/minislam/actions/workflows/ci.yml)

Simple implementation of monocular visual odometry with loop closure detection.

Rewrite here: https://github.com/markoelez/minislam2.

### Features
- ORB feature detection and matching
- Essential matrix decomposition for pose estimation
- **Loop closure detection** - detects when revisiting previously seen locations
- Real-time 3D visualization with loop closure visualization (yellow lines)

![alt text](https://github.com/markoelez/minislam/blob/master/img/example.png?raw=true)
![alt text](https://github.com/markoelez/minislam/blob/master/img/example_two.png?raw=true)


### Usage

run the following for a quick example:
```sh
uv run main
```

using your own data:
1. start by doing one of the following:
    - download a sequence of images: https://www.cvlibs.net/datasets/kitti/raw_data.php
    - download an MP4 video
2. move the dataset into the `eval/` directory
3. add dataset metadata to the `config.yaml` file
4. run with `uv run main --dataset=[your_dataset_name]`


### Development

Install pre-commit hooks:
```sh
uvx pre-commit install
```

Run linting and formatting:
```sh
uvx ruff check .     # linting
uvx ruff format .    # formatting
```

Run type checking:
```sh
uvx ty check
```

#### Todo
- add tests
- add pose graph optimization (correct drift using loop closures)
- test out other feature detectors (SIFT, SuperPoint)
