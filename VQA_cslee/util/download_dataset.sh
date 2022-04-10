#!/bin/tcsh

#########################################################

# One may need to change directory for datasets like this.
#set DATASETS_DIR = "/run/media/hoosiki/WareHouse3/mtb/datasets/VQA"

mkdir -p "../datasets"
DATASETS_DIR="../datasets"

##########################################################

ANNOTATIONS_DIR="$DATASETS_DIR/Annotations"
QUESTIONS_DIR="$DATASETS_DIR/Questions"
IMAGES_DIR="$DATASETS_DIR/Images"


wget -O $ANNOTATIONS_DIR/v2_Annotations_Train_mscoco.zip "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
wget -O $ANNOTATIONS_DIR/v2_Annotations_Val_mscoco.zip "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"

##########################################################

unzip $ANNOTATIONS_DIR/v2_Annotations_Train_mscoco.zip -d $ANNOTATIONS_DIR
unzip $ANNOTATIONS_DIR/v2_Annotations_Val_mscoco.zip -d $ANNOTATIONS_DIR

unzip $QUESTIONS_DIR/v2_Questions_Train_mscoco.zip -d $QUESTIONS_DIR
unzip $QUESTIONS_DIR/v2_Questions_Val_mscoco.zip -d $QUESTIONS_DIR
unzip $QUESTIONS_DIR/v2_Questions_Test_mscoco.zip -d $QUESTIONS_DIR

unzip $IMAGES_DIR/train2014.zip -d $IMAGES_DIR
unzip $IMAGES_DIR/val2014.zip -d $IMAGES_DIR
unzip $IMAGES_DIR/test2015.zip -d $IMAGES_DIR
