# caffe-binary-upgrade
caffe-binary-upgrade is a tool which takes a caffemodel as an input that is defined with V1LayerParameters.
This converts into a caffemodel that is upgraded to LayerParameters.

### Example

Download the VGG-Net 19 Layer caffemodel from the following site which is trained with V1LayerParameters [here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)

### Usage:
```
    % upgrade_layer_parameters VGG_ILSVRC_19_layers.caffemodel
```
This automatically upgrades the caffemodel and writes into a new caffemodel named net.caffemodel which contains LayerParameters.





