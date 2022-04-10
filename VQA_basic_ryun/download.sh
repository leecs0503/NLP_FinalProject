#!/bin/bash

#########################################################

mkdir -p "./datasets"

# Download datasets from VQA official url: https://visualqa.org/download.html

# VQA Annotations
mkdir -p "./datasets/Annotations"
ANNOTATIONS_DIR=./datasets/Annotations
URL_TRAIN="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
URL_VAL="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
FILE_TRAIN=./datasets/Annotations/v2_Annotations_Train_mscoco.zip
FILE_VAL=./datasets/Annotations/v2_Annotations_Val_mscoco.zip

wget -N $URL_TRAIN -O $FILE_TRAIN
wget -N $URL_VAL -O $FILE_VAL

unzip $FILE_TRAIN -d $ANNOTATIONS_DIR
unzip $FILE_VAL -d $ANNOTATIONS_DIR

rm $FILE_TRAIN
rm $FILE_VAL

# VQA Input Questions
mkdir -p "./datasets/Questions"
QUESTIONS_DIR=./datasets/Questions
URL_TRAIN="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
URL_VAL="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
URL_TEST="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
FILE_TRAIN=./datasets/Questions/v2_Questions_Train_mscoco.zip 
FILE_VAL=./datasets/Questions/v2_Questions_Val_mscoco.zip 
FILE_TEST=./datasets/Questions/v2_Questions_Test_mscoco.zip

wget -N $URL_TRAIN -O $FILE_TRAIN
wget -N $URL_VAL -O $FILE_VAL
wget -N $URL_TEST -O $FILE_TEST

unzip $FILE_TRAIN -d $QUESTIONS_DIR
unzip $FILE_VAL -d $QUESTIONS_DIR
unzip $FILE_TEST -d $QUESTIONS_DIR

rm $FILE_TRAIN
rm $FILE_VAL
rm $FILE_TEST

# VQA Input Images (COCO)
mkdir -p "./datasets/Images"
IMAGES_DIR=./datasets/Images
URL_TRAIN="http://images.cocodataset.org/zips/train2014.zip"
URL_VAL="http://images.cocodataset.org/zips/val2014.zip"
URL_TEST="http://images.cocodataset.org/zips/test2015.zip"
FILE_TRAIN=./datasets/Images/train2014.zip
FILE_VAL=./datasets/Images/val2014.zip
FILE_TEST=./datasets/Images/test2015.zip

wget -N $URL_TRAIN -O $FILE_TRAIN
wget -N $URL_VAL -O $FILE_VAL
wget -N $URL_TEST -O $FILE_TEST

unzip $FILE_TRAIN -d $IMAGES_DIR
unzip $FILE_VAL -d $IMAGES_DIR
unzip $FILE_TEST -d $IMAGES_DIR

rm $FILE_TRAIN
rm $FILE_VAL
rm $FILE_TEST