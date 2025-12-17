# minislam

Simple implementation of monocular visual odometry.

Rewrite here: https://github.com/markoelez/minislam2.

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
- add graph optimization
- test out other feature detectors
