#!/bin/bash
# Run the program "bash download_dataset.sh celex" or "bash download_dataset.sh cmudict"

if [[ $1 == "celex" ]];
then
	DIR=./celex

	# wget output file
	FILE="celex.txt"

	# wget URL
	URL=https://raw.githubusercontent.com/pradiptadeb1990/data/master/celex_g2p.dict?token=AJs_5fGzCP2xcW3XsaVWowtmTuz7-Vbwks5asmSPwA%3D%3D
elif [[ $1 == "cmudict" ]];
then
   	DIR=./cmudict

	# wget output file
	FILE="cmudict.txt"

	# wget URL
	URL=https://raw.githubusercontent.com/pradiptadeb1990/data/master/cmudict_old.dict?token=AJs_5ZlDMMy8X4s3qSth2KCY6zpfZgkHks5asmSRwA%3D%3D
else
	echo "Please use either 'cmudict' or 'celex' as argument"
	exit 1
fi;

cd $DIR
wget $URL -O $FILE