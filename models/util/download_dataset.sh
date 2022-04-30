#!/bin/sh

##############################################################################


DATASETS_DIR="../datasets"

echo "mkdir $DATASETS_DIR"
mkdir -p $DATASETS_DIR

##############################################################################

ANNOTATIONS_DIR="$DATASETS_DIR/Annotations"
QUESTIONS_DIR="$DATASETS_DIR/Questions"
IMAGES_DIR="$DATASETS_DIR/Images"
Complementary_DIR="$DATASETS_DIR/Complementary"

echo "##############################################################################"
echo "ANNOTATIONS_DIR = $ANNOTATIONS_DIR"
echo "QUESTIONS_DIR = $QUESTIONS_DIR"
echo "IMAGES_DIR = $IMAGES_DIR"
echo "Complementary = $Complementary_DIR"
mkdir -p $ANNOTATIONS_DIR
mkdir -p $QUESTIONS_DIR
mkdir -p $IMAGES_DIR
mkdir -p $Complementary_DIR
echo "##############################################################################"

##############################################################################


# Download datasets from VQA official url: https://visualqa.org/download.html

# VQA Annotations
echo "Download Train & Validation Annotations"
wget -O $ANNOTATIONS_DIR/v2_Annotations_Train_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
wget -O $ANNOTATIONS_DIR/v2_Annotations_Val_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"


# VQA Input Questions
echo "Download Train & Validation Questions"
wget -O $QUESTIONS_DIR/v2_Questions_Train_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
wget -O $QUESTIONS_DIR/v2_Questions_Val_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
wget -O $QUESTIONS_DIR/v2_Questions_Test_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"

# VQA Input Images (COCO)
echo "Download Train & Validation Images"
wget -O $IMAGES_DIR/train2014.zip -nc "http://images.cocodataset.org/zips/train2014.zip"
wget -O $IMAGES_DIR/val2014.zip -nc "http://images.cocodataset.org/zips/val2014.zip"
wget -O $IMAGES_DIR/test2015.zip -nc "http://images.cocodataset.org/zips/test2015.zip"

# VQA Input Complementary Pairs List
echo "Download Train & Validation Images"
wget -O $Complementary_DIR/v2_Complementary_Pairs_Train_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip"
wget -O $Complementary_DIR/v2_Complementary_Pairs_Val_mscoco.zip -nc "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip"

##########################################################

echo "unzip start"
unzip $ANNOTATIONS_DIR/v2_Annotations_Train_mscoco.zip -d $ANNOTATIONS_DIR
unzip $ANNOTATIONS_DIR/v2_Annotations_Val_mscoco.zip -d $ANNOTATIONS_DIR

unzip $QUESTIONS_DIR/v2_Questions_Train_mscoco.zip -d $QUESTIONS_DIR
unzip $QUESTIONS_DIR/v2_Questions_Val_mscoco.zip -d $QUESTIONS_DIR
unzip $QUESTIONS_DIR/v2_Questions_Test_mscoco.zip -d $QUESTIONS_DIR

unzip $IMAGES_DIR/train2014.zip -d $IMAGES_DIR
unzip $IMAGES_DIR/val2014.zip -d $IMAGES_DIR
unzip $IMAGES_DIR/test2015.zip -d $IMAGES_DIR

unzip $Complementary_DIR/v2_Complementary_Pairs_Train_mscoco.zip -d $Complementary_DIR
unzip $Complementary_DIR/v2_Complementary_Pairs_Val_mscoco.zip -d $Complementary_DIR

