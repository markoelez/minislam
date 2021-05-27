#!/bin/bash
mkdir build
cd build

git clone --recursive git@github.com:markoelez/Pangolin.git pangolin
cd pangolin
mkdir build
cd build

cmake .. -DPYBIND11_PYTHON_VERSION=3.9 -DBUILD_PANGOLIN_LIBREALSENSE=OFF

make -j8
