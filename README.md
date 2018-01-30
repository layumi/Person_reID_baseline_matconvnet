# Person_reID_baseline_matconvnet
Matconvnet implement of Person re-identification baseline. We arrived Rank@1=87.74% mAP=69.46% only with softmax loss.

## Installation
1. Clone this repo
 
	```Shell
	git clone https://github.com/layumi/Person_reID_baseline_matconvnet.git
	cd Person_reID_baseline_matconvnet
	mkdir data
	```
	
2. Compile matconvnet 

	You just need to uncomment and modify some lines in `gpu_compile.m` and run it in Matlab. Try it~

	If you fail in compilation, you may refer to http://www.vlfeat.org/matconvnet/install/

## Test 
1. After installation, you can run `test/test_duke.m` to extract the features of images in the gallery and query set. They will store in a .mat file. Then you can use it to do evaluation.

2. 


## Train
1. Add your dataset path into `prepare_data.m` and run it. Make sure the code outputs the right image path.

2. Run `train_id_net_.m` to have fun.

