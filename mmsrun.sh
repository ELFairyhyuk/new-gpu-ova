#!/usr/bin/env bash
trainingData="dataset/shuttle.scale dataset/usps dataset/poker dataset/pendigits dataset/mnist.scale dataset/satimage.scale dataset/letter.scale dataset/acoustic_scale dataset/seismic.scale" # sector news20 shuttle
#trainingData="dataset/satimage.scale dataset/shuttle.scale "
#filename="dataset/poker"
for filename in $trainingData
do
	testfilename=${filename}".t"
	C=1
	gamma=0.1
			./bin/release/mascot -b 0 -o 6 -c $C -g $gamma -r 1 -e $testfilename $filename ;
done
