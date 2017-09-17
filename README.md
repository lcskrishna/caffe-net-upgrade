# caffe-binary-upgrade
caffe-binary-upgrade is a tool which takes a caffemodel or a prototxt file as an input that is defined with V1LayerParameters. (Caffe deprecated definition) . 
This implementation upgrades to the layer parameters. 

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

### Usage:
```
    % upgrade_layer_parameters <net.caffemodel | net.prototxt> [output_file_prefix]
```
### Example 1 : Upgrade VGG caffemodel

Download the VGG-Net 19 Layer caffemodel from the following site which is trained with V1LayerParameters [here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)

Execute the following command : 
```
    % upgrade_layer_parameters VGG.caffemodel output
```
This upgrades the caffemodel and writes into a new caffemodel named output.caffemodel.

### Example 2 : Upgrade VGG prototxt

Download the VGG-Net 19 Layers prototxt defenition from the following site defined with V1LayerParameters [here](https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt)

Execute the following command:
```
    % upgrade_layer_parameters vgg.prototxt output
```

This upgrades the prototxt defenition and writes into a new file named output.prototxt




