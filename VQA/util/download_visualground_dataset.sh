
DATASETS_DIR="../datasets/visual_ground"

echo "mkdir $DATASETS_DIR"
mkdir -p $DATASETS_DIR

##############################################################################

wget -O $DATASETS_DIR/refcocog.zip -nc "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"

unzip $DATASETS_DIR/refcocog.zip -d $DATASETS_DIR