for j in 1 2 3 4 5 6 7 8 9 10
do
	python train.py -d URBAN_SED -f MelSpectrogram -m AttProtos -fold test
	mv URBAN_SED/AttProtos/test URBAN_SED/AttProtos/test_$j
	
	python train.py -d URBAN_SED -f Openl3 -m MLP -fold test
	mv URBAN_SED/MLP/test URBAN_SED/MLP/test_$j
	
	python train.py -d URBAN_SED -f MelSpectrogram -m SB_CNN_SED -fold test
	mv URBAN_SED/SB_CNN_SED/test URBAN_SED/SB_CNN_SED/test_$j

done
