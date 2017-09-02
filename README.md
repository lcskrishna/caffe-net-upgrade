# caffe-binary-upgrade
caffe-binary-upgrade is a tool which takes a caffemodel as an input that is defined with V1LayerParameters.
This converts into a caffemodel that is upgraded to LayerParameters.

### Pre-requisites
1. Ubuntu 16.04
2. CMake 2.8 or newer [download](https://cmake.org/download/)
3. Install the [Protobuf](https://github.com/google/protobuf) library.

### Build Instructions
1. Install the respective pre-requisites if not present. Make sure to install libprotobuf-dev and protobuf-compiler
2. mkdir build
3. cd build
4. cmake ../caffe-binary-upgrade .
5. make

This automatically builds a executable in the build folder. 

### Example

Download the VGG-Net 19 Layer caffemodel from the following site which is trained with V1LayerParameters [here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)

### Usage:
```
    % upgrade_layer_parameters VGG_ILSVRC_19_layers.caffemodel
```
This automatically upgrades the caffemodel and writes into a new caffemodel named net.caffemodel which contains LayerParameters.





