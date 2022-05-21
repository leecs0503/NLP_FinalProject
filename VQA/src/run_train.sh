#!/bin/bash

for ((step_size=9; step_size<=11; step_size+=2))
do
for learning_rate in 0.01 0.001 0.0001
do
for dropout_rate in 0.4 0.5 0.6
do
for gamma in 0.1 0.3 0.5
do
    python src/train.py --step_size=${step_size} --learning_rate=${learning_rate} --dropout_rate=${dropout_rate} --gamma=${gamma} --num_epochs=6 --tensorboard_dir="runs/adam|${step_size}|${learning_rate}|${dropout_rate}|${gamma}"
done
done
done
done

for ((step_size=9; step_size<=11; step_size+=2))
do
for learning_rate in 0.01 0.001 0.0001
do
for momentum in 0.9 0.95 0.99
do
for dropout_rate in 0.4 0.5 0.6
do
for gamma in 0.1 0.3 0.5
do
    python src/train.py --step_size=${step_size} --learning_rate=${learning_rate} --momentum=${momentum} --dropout_rate=${dropout_rate} --gamma=${gamma} --num_epochs=6 --tensorboard_dir="runs/SGD|${step_size}|${learning_rate}|${momentum}|${dropout_rate}|${gamma}"
done
done
done
done 
done