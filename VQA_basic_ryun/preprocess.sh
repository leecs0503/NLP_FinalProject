#!/bin/bash

python utils/build_vqa_inputs.py
python utils/make_vocabs4QA.py
python utils/resize_images.py
