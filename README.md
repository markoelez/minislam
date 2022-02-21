#### minislam

Simple implementation of monocular visual odometry.

![alt text](https://github.com/markoelez/minislam/blob/master/example.png?raw=true)
![alt text](https://github.com/markoelez/minislam/blob/master/example_two.png?raw=true)


usage
-----
example:
`./main.py --config_section=kitti2`

Add new configurations to `config.ini` file as new sections. The `path` parameter can either be a directory of images or a video file.

todo
-----
- cleanup code
- add tests
- parameterize camera intrinsics
- add graph optimization
- test out other feature detectors
