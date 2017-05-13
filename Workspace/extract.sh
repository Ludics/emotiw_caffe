#!/bin/bash
cd preprocess
python face_landmark_detection_xf_2.py shape_predictor_68_face_landmarks.dat $1
for file in $(ls $1/*.jpg)
do
	./RightPosition_final $file
done
rm ${1}/*.txt
#RightPosition_try_2 $1
