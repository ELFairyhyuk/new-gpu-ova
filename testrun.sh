#!/usr/bin/env bash
trainingData="dataset/rcv1" # sector news20 shuttle
#trainingData="dataset/satimage.scale dataset/shuttle.scale "
#filename="dataset/poker"
for filename in $trainingData
do
	testfilename=${filename}".t"
	C=0.5
		echo $C
	for (( i=0; i<1;i++ ))
	do
		#C=$(echo "scale=10; $C/2 "|bc)  
		C=$(echo "2*$C"|bc)
		gamma=256
		for(( j=0; j<1; j++))
		do
			gamma=$(echo "scale=10; $gamma / 2 "|bc)
			echo $gamma
			echo $testfilename
			./bin/release/mascot -b 0 -o 5 -c $C -g $gamma -r 1 -e $testfilename $filename ;
		done
	done
done
