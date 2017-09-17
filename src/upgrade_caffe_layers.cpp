/*
MIT License

Copyright (c) 2017 Chaitanya Sri Krishna Lolla

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "caffe.pb.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

using namespace caffe;

//map v1 layer types to layer types.
const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type) {
	switch (type) {
		case V1LayerParameter_LayerType_NONE:
			return "";
		case V1LayerParameter_LayerType_ABSVAL:
			return "AbsVal";
		case V1LayerParameter_LayerType_ACCURACY:
			return "Accuracy";
		case V1LayerParameter_LayerType_ARGMAX:
			return "ArgMax";
		case V1LayerParameter_LayerType_BNLL:
			return "BNLL";
		case V1LayerParameter_LayerType_CONCAT:
			return "Concat";
		case V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
			return "ContrastiveLoss";
		case V1LayerParameter_LayerType_CONVOLUTION:
			return "Convolution";
		case V1LayerParameter_LayerType_DECONVOLUTION:
			return "Deconvolution";
		case V1LayerParameter_LayerType_DATA:
			return "Data";
		case V1LayerParameter_LayerType_DROPOUT:
			return "Dropout";
		case V1LayerParameter_LayerType_DUMMY_DATA:
			return "DummyData";
		case V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
			return "EuclideanLoss";
		case V1LayerParameter_LayerType_ELTWISE:
			return "Eltwise";
		case V1LayerParameter_LayerType_EXP:
			return "Exp";
		case V1LayerParameter_LayerType_FLATTEN:
			return "Flatten";
		case V1LayerParameter_LayerType_HDF5_DATA:
			return "HDF5Data";
		case V1LayerParameter_LayerType_HDF5_OUTPUT:
			return "HDF5Output";
		case V1LayerParameter_LayerType_HINGE_LOSS:
			return "HingeLoss";
		case V1LayerParameter_LayerType_IM2COL:
			return "Im2col";
		case V1LayerParameter_LayerType_IMAGE_DATA:
			return "ImageData";
		case V1LayerParameter_LayerType_INFOGAIN_LOSS:
			return "InfogainLoss";
		case V1LayerParameter_LayerType_INNER_PRODUCT:
			return "InnerProduct";
		case V1LayerParameter_LayerType_LRN:
			return "LRN";
		case V1LayerParameter_LayerType_MEMORY_DATA:
			return "MemoryData";
		case V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
			return "MultinomialLogisticLoss";
		case V1LayerParameter_LayerType_MVN:
			return "MVN";
		case V1LayerParameter_LayerType_POOLING:
			return "Pooling";
		case V1LayerParameter_LayerType_POWER:
			return "Power";
		case V1LayerParameter_LayerType_RELU:
			return "ReLU";
		case V1LayerParameter_LayerType_SIGMOID:
			return "Sigmoid";
		case V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
			return "SigmoidCrossEntropyLoss";
		case V1LayerParameter_LayerType_SILENCE:
			return "Silence";
		case V1LayerParameter_LayerType_SOFTMAX:
			return "Softmax";
		case V1LayerParameter_LayerType_SOFTMAX_LOSS:
			return "SoftmaxWithLoss";
		case V1LayerParameter_LayerType_SPLIT:
			return "Split";
		case V1LayerParameter_LayerType_SLICE:
			return "Slice";
		case V1LayerParameter_LayerType_TANH:
			return "TanH";
		case V1LayerParameter_LayerType_WINDOW_DATA:
			return "WindowData";
		case V1LayerParameter_LayerType_THRESHOLD:
			return "Threshold";
		default:
			std::cerr << "ERROR: Unknown V1LayerParameter layer type: " << type << std::endl;

		return "";
	}
}

//upgrade v1 layer parameters to layer parameters.
bool upgrade_v1_layer_parameters(const caffe::V1LayerParameter& v1_layer_param, caffe::LayerParameter * layer_param)
{

	layer_param->Clear();
	bool is_fully_compatible = true;
	for (int i = 0; i < v1_layer_param.bottom_size(); ++i) {
		layer_param->add_bottom(v1_layer_param.bottom(i));
	}
	for (int i = 0; i < v1_layer_param.top_size(); ++i) {
		layer_param->add_top(v1_layer_param.top(i));
	}
	if (v1_layer_param.has_name()) {
		layer_param->set_name(v1_layer_param.name());
	}
	for (int i = 0; i < v1_layer_param.include_size(); ++i) {
		layer_param->add_include()->CopyFrom(v1_layer_param.include(i));
	}
	for (int i = 0; i < v1_layer_param.exclude_size(); ++i) {
		layer_param->add_exclude()->CopyFrom(v1_layer_param.exclude(i));
	}
	if (v1_layer_param.has_type()) {
		layer_param->set_type(UpgradeV1LayerType(v1_layer_param.type()));
	}
	for (int i = 0; i < v1_layer_param.blobs_size(); ++i) {
		layer_param->add_blobs()->CopyFrom(v1_layer_param.blobs(i));
	}
	for (int i = 0; i < v1_layer_param.param_size(); ++i) {
		while (layer_param->param_size() <= i) { layer_param->add_param(); }
		layer_param->mutable_param(i)->set_name(v1_layer_param.param(i));
	}
	ParamSpec_DimCheckMode mode;
	for (int i = 0; i < v1_layer_param.blob_share_mode_size(); ++i) {
		while (layer_param->param_size() <= i) { layer_param->add_param(); }
		switch (v1_layer_param.blob_share_mode(i)) {
			case V1LayerParameter_DimCheckMode_STRICT:
				mode = ParamSpec_DimCheckMode_STRICT;
				break;
			case V1LayerParameter_DimCheckMode_PERMISSIVE:
				mode = ParamSpec_DimCheckMode_PERMISSIVE;
				break;
			default:
				std::cerr << "ERROR: Unknown blob_share_mode: "
					<< v1_layer_param.blob_share_mode(i) << std::endl;;
				break;
		}
		layer_param->mutable_param(i)->set_share_mode(mode);
	}
	for (int i = 0; i < v1_layer_param.blobs_lr_size(); ++i) {
		while (layer_param->param_size() <= i) { layer_param->add_param(); }
		layer_param->mutable_param(i)->set_lr_mult(v1_layer_param.blobs_lr(i));
	}
	for (int i = 0; i < v1_layer_param.weight_decay_size(); ++i) {
		while (layer_param->param_size() <= i) { layer_param->add_param(); }
		layer_param->mutable_param(i)->set_decay_mult(
				v1_layer_param.weight_decay(i));
	}
	for (int i = 0; i < v1_layer_param.loss_weight_size(); ++i) {
		layer_param->add_loss_weight(v1_layer_param.loss_weight(i));
	}
	if (v1_layer_param.has_accuracy_param()) {
		layer_param->mutable_accuracy_param()->CopyFrom(
				v1_layer_param.accuracy_param());
	}
	if (v1_layer_param.has_argmax_param()) {
		layer_param->mutable_argmax_param()->CopyFrom(
				v1_layer_param.argmax_param());
	}
	if (v1_layer_param.has_concat_param()) {
		layer_param->mutable_concat_param()->CopyFrom(
				v1_layer_param.concat_param());
	}
	if (v1_layer_param.has_contrastive_loss_param()) {
		layer_param->mutable_contrastive_loss_param()->CopyFrom(
				v1_layer_param.contrastive_loss_param());
	}
	if (v1_layer_param.has_convolution_param()) {
		layer_param->mutable_convolution_param()->CopyFrom(
				v1_layer_param.convolution_param());
	}
	if (v1_layer_param.has_data_param()) {
		layer_param->mutable_data_param()->CopyFrom(
				v1_layer_param.data_param());
	}
	if (v1_layer_param.has_dropout_param()) {
		layer_param->mutable_dropout_param()->CopyFrom(
				v1_layer_param.dropout_param());
	}
	if (v1_layer_param.has_dummy_data_param()) {
		layer_param->mutable_dummy_data_param()->CopyFrom(
				v1_layer_param.dummy_data_param());
	}
	if (v1_layer_param.has_eltwise_param()) {
		layer_param->mutable_eltwise_param()->CopyFrom(
				v1_layer_param.eltwise_param());
	}
	if (v1_layer_param.has_exp_param()) {
		layer_param->mutable_exp_param()->CopyFrom(
				v1_layer_param.exp_param());
	}
	if (v1_layer_param.has_hdf5_data_param()) {
		layer_param->mutable_hdf5_data_param()->CopyFrom(
				v1_layer_param.hdf5_data_param());
	}
	if (v1_layer_param.has_hdf5_output_param()) {
		layer_param->mutable_hdf5_output_param()->CopyFrom(
				v1_layer_param.hdf5_output_param());
	}
	if (v1_layer_param.has_hinge_loss_param()) {
		layer_param->mutable_hinge_loss_param()->CopyFrom(
				v1_layer_param.hinge_loss_param());
	}
	if (v1_layer_param.has_image_data_param()) {
		layer_param->mutable_image_data_param()->CopyFrom(
				v1_layer_param.image_data_param());
	}
	if (v1_layer_param.has_infogain_loss_param()) {
		layer_param->mutable_infogain_loss_param()->CopyFrom(
				v1_layer_param.infogain_loss_param());
	}
	if (v1_layer_param.has_inner_product_param()) {
		layer_param->mutable_inner_product_param()->CopyFrom(
				v1_layer_param.inner_product_param());
	}
	if (v1_layer_param.has_lrn_param()) {
		layer_param->mutable_lrn_param()->CopyFrom(
				v1_layer_param.lrn_param());
	}
	if (v1_layer_param.has_memory_data_param()) {
		layer_param->mutable_memory_data_param()->CopyFrom(
				v1_layer_param.memory_data_param());
	}
	if (v1_layer_param.has_mvn_param()) {
		layer_param->mutable_mvn_param()->CopyFrom(
				v1_layer_param.mvn_param());
	}
	if (v1_layer_param.has_pooling_param()) {
		layer_param->mutable_pooling_param()->CopyFrom(
				v1_layer_param.pooling_param());
	}
	if (v1_layer_param.has_power_param()) {
		layer_param->mutable_power_param()->CopyFrom(
				v1_layer_param.power_param());
	}
	if (v1_layer_param.has_relu_param()) {
		layer_param->mutable_relu_param()->CopyFrom(
				v1_layer_param.relu_param());
	}
	if (v1_layer_param.has_sigmoid_param()) {
		layer_param->mutable_sigmoid_param()->CopyFrom(
				v1_layer_param.sigmoid_param());
	}
	if (v1_layer_param.has_softmax_param()) {
		layer_param->mutable_softmax_param()->CopyFrom(
				v1_layer_param.softmax_param());
	}
	if (v1_layer_param.has_slice_param()) {
		layer_param->mutable_slice_param()->CopyFrom(
				v1_layer_param.slice_param());
	}
	if (v1_layer_param.has_tanh_param()) {
		layer_param->mutable_tanh_param()->CopyFrom(
				v1_layer_param.tanh_param());
	}
	if (v1_layer_param.has_threshold_param()) {
		layer_param->mutable_threshold_param()->CopyFrom(
				v1_layer_param.threshold_param());
	}
	if (v1_layer_param.has_window_data_param()) {
		layer_param->mutable_window_data_param()->CopyFrom(
				v1_layer_param.window_data_param());
	}
	if (v1_layer_param.has_transform_param()) {
		layer_param->mutable_transform_param()->CopyFrom(
				v1_layer_param.transform_param());
	}
	if (v1_layer_param.has_loss_param()) {
		layer_param->mutable_loss_param()->CopyFrom(
				v1_layer_param.loss_param());
	}
	if (v1_layer_param.has_layer()) {
		std::cerr << "ERROR: Input NetParameter has V0 layer -- ignoring upgrade." << std::endl;;
		is_fully_compatible = false;
	}

	return is_fully_compatible;

}

void check_network_details(const caffe::NetParameter& net_parameter, caffe::NetParameter * upgraded_net_param)
{
	if(net_parameter.has_name())
		std::cout << "INFO: Network loaded is : " << net_parameter.name() << std::endl;

	int layer_count = net_parameter.layer_size();
	if(layer_count > 0) {
		std::cout << "INFO: This network doesn't require upgrade" << std::endl;
		exit(1);
	}
	else {
		int v1_layer_count = net_parameter.layers_size();
		if(v1_layer_count > 0) {
			std::cout << "INFO: Upgrading V1LayerParameter => LayerParameter" << std::endl;
			upgraded_net_param->CopyFrom(net_parameter);
			upgraded_net_param->clear_layer();
			upgraded_net_param->clear_layers();
			for(int i=0 ; i < v1_layer_count ; i++) {
				bool isUpgraded = upgrade_v1_layer_parameters(net_parameter.layers(i), upgraded_net_param->add_layer());
				if(!isUpgraded) {
					std::cerr << "ERROR: Unable to upgrade, make sure caffemodel has V1LayerParameters." << std::endl;
					exit(1);
				}
			} 
		} 
		else{
			std::cerr << "ERROR: Unsupported layer type for upgrade" << std::endl;
			exit(1);
		}
	}
}

void loadCaffeModel(const char * fileName , std::string& output_file_prefix)
{
	//verify the version of protobuf library
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	//read the caffemodel.
	caffe::NetParameter net_parameter;
	std::cout << "INFO: Reading the binary file from : " << fileName << std::endl;
	{
		std::fstream input(fileName, std::ios::in | std::ios::binary);
		bool isSuccess = net_parameter.ParseFromIstream(&input);
		if(isSuccess) {
			std::cout << "INFO: Successfully read caffemodel" << std::endl;
			caffe::NetParameter upgraded_net_parameter;
			check_network_details(net_parameter, &upgraded_net_parameter);
			std::cout << "STATUS: Upgrade Successful" << std::endl;
			
			//write updated caffemodel.
			std::string out_file = output_file_prefix + ".caffemodel";
			std::fstream output(out_file.c_str(), std::ios::out | std::ios::binary);
			if(!upgraded_net_parameter.SerializeToOstream(&output)) {
				std::cerr << "ERROR: Unable to write the upgraded caffemodel." << std::endl;
			}
			else {
				std::cout << "INFO: upgraded file written successfully into " << out_file << std::endl;
			}
		}
		else {
			std::cerr << "ERROR: Unable to parse caffemodel" << std::endl;
			exit(1);
		}
	}

}

void removeUnknownTypes(std::string& str, const std::string& from, const std::string& to){
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
	str.replace(start_pos, from.length(), to);
	start_pos += to.length(); 
    }
}

void loadPrototxt(const char * fileName , std::string& output_file_prefix)
{
	//verify the version of protobuf library
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	//read the caffemodel.
	caffe::NetParameter net_parameter;
	std::cout << "INFO: Reading the prototxt file from : " << fileName << std::endl;
	//Read the Prototxt File.
    {
	int fd = open(fileName,O_RDONLY);
	if(fd < 0) {
		std::cerr << "ERROR: Unable to open the file : " << fileName << std::endl;
	}

	google::protobuf::io::FileInputStream fi(fd);
	fi.SetCloseOnDelete(true);
	if(!google::protobuf::TextFormat::Parse(&fi, &net_parameter)) {
		std::cerr << "ERROR: Failed to parse the file : " << fileName << std::endl;
		exit(1);
	}
	else {
		std::cout << "INFO: prototxt read successful " << std::endl;
		//upgrade prototxt if needed.
		caffe::NetParameter upgrade_net_parameter;
		check_network_details(net_parameter, &upgrade_net_parameter);
		std::cout << "STATUS: upgrade successful. " << std::endl;

		//write defenition into a prototxt file.
		std::fstream fs;
		std::string out_file = output_file_prefix + ".prototxt";
		fs.open(out_file.c_str(), std::ios::out);
		std::string out;
		
		if(google::protobuf::TextFormat::PrintToString(upgrade_net_parameter, &out)) {
			std::cout << "INFO: upgraded net is written into " << out_file << std::endl;
			removeUnknownTypes(out, "95:0", "");
			fs << out ; 
		}
		else {
			std::cout << "ERROR: Unable to write upgraded net to file " << std::endl;
			exit(1);
		}
	}
    }	
}

int main(int argc, char * argv[])
{
	const char * usage = "Usage: upgrade_layer_parameters <net.caffemodel | net.prototxt> [output_file_prefix]";

	//get options.
	if(argc < 2 ) {
		printf("%s\n",usage);
		return -1;
	}

	const char * fileName = argv[1];
	std::string output_file_prefix = "net";
	if(argc > 2) output_file_prefix = argv[2];

	if(strstr(fileName,".caffemodel")) {
		std::cout << "Loading caffemodel file ... " << std::endl;
		loadCaffeModel(fileName , output_file_prefix);
	}
	else if(strstr(fileName, ".prototxt")) {
		std::cout << "Loading prototxt file ... " << std::endl;
		loadPrototxt(fileName , output_file_prefix);
	}

	return 0;
}
