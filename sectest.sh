#!/usr/bin/env bash
#trainingData="dataset/letter.scale dataset/satimage.scale dataset/usps dataset/acoustic_scale "
filename="dataset/poker"
#for filename in $trainingData
#do
	testfilename=${filename}".t"
	C=512
	for (( i=0; i<1;i++ ))
	do
		C=$(echo "2*$C"|bc)
		gamma=2048
		for(( j=0; j<30; j++))
		do
			gamma=$(echo "scale=10; $gamma / 2 "|bc)
			echo $gamma
			echo $testfilename
			./bin/release/mascot -b 0 -o 5 -c $C -g $gamma -r 1 -e $testfilename $filename ;
		done
	done
#done
