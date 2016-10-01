#import files
import numpy as np
from os import listdir
import cv2
from os.path import isfile, join
import sys

#Initailize Network
def createNet(caffe):
	model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
	model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
	net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)
	net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)
	return net

#Setting Caffe Solver Mode
def setCaffeMode(caffe,x):
	if(x==1):
		caffe.set_device(0)  # if we have multiple GPUs, pick the first one
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	
	return caffe

#creating transformer for transforming the image to required type 
def createTransformer(caffe,net,caffe_root):
	#loading the mean value
	mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1)
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))
	return transformer

# Input Path of directory, numpy array, list of class
def createFeature(caffe_root,caffe,directory,label):
	folderPath=caffe_root+directory
	onlyfiles = [ f for f in listdir(folderPath) if isfile(join(folderPath,f)) ]
	class_label=[]
	feature_array=np.zeros((len(onlyfiles),4096),dtype=np.float32)
	
	for i in range(0,len(onlyfiles)):
		image = caffe.io.load_image(join(folderPath,onlyfiles[i]))
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image
		output = net.forward()
		flatten_feature = net.blobs['fc7'].data[0]
		feature=flatten_feature.reshape(1,4096)
		arr=np.array(feature,dtype=np.float32)
		feature_array[i]=arr
		class_label+=[int(label)]
	
	return feature_array,class_label
	
	

#Initailizing Caffe and the model parameters

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
#Set Caffe mode to GPU
caffe=setCaffeMode(caffe,int(1))

#Create Net along with strusture and weights
net=createNet(caffe)

#create transformer of Image
transformer=createTransformer(caffe,net,caffe_root)



print "Model Loaded successfully"
#getting Image Feature
feature_array_horse,label_horse=createFeature(caffe_root,caffe,'examples/ToUpload/Horses/Train',0)   
print "Horse done"                 
feature_array_bike,label_bike=createFeature(caffe_root,caffe,'examples/ToUpload/Bikes/Train',1)                    
print "Bike done"                          
print feature_array_horse.shape


np.save("featureData/HorseFeature", feature_array_horse)
np.save("featureData/BikeFeature", feature_array_bike)
np.save("featureData/HorseLabel", np.array(label_horse))
np.save("featureData/BikeLabel", np.array(label_bike))                          



