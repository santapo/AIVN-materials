gdrive download 1G5F10s16MNWBoqPuxH0gggb41chlkZQU 
unzip -qq birds_400.zip -d data
rm birds_400.zip

mkdir data/lvl_classes
cp -r data/little_classes/lvl_train data/lvl_classes/train
cp -r data/little_classes/valid data/lvl_classes/valid
cp -r data/little_classes/test data/lvl_classes/test

mkdir data/ll_classes
cp -r data/little_classes/ll_train data/ll_classes/train
cp -r data/little_classes/valid data/ll_classes/valid
cp -r data/little_classes/test data/ll_classes/test

mkdir data/ml_classes
cp -r data/many_classes/ml_train data/ml_classes/train
cp -r data/many_classes/valid data/ml_classes/valid
cp -r data/many_classes/test data/ml_classes/test

mkdir data/mvl_classes
cp -r data/many_classes/mvl_train data/mvl_classes/train
cp -r data/many_classes/valid data/mvl_classes/valid
cp -r data/many_classes/test data/mvl_classes/test

gdrive download 1sYTrinRuZLM9Pxzv4BzvDjDHhZW7M62Q
unzip -qq chest_xray.zip -d data
rm chest_xray.zip