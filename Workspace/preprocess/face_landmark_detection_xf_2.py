#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys
import os
import dlib
import glob
import numpy
import cv2

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_landmarks(im):
    rects = detector(im, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
detectedNum = 0
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    (pathname,filename)=os.path.split(f)
    outputname=filename[:-4]

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) > 0 :

		if (dets[0].bottom()-dets[0].top()<100 and len(dets)>1) or (dets[0].bottom()-dets[0].top()<50) :
			os.remove(f)
			print("Deleted")

		else :
			shape = predictor(img, dets[0])
			#print("face_landmark:")
			#print(get_landmarks(img))
			rects = detector(img, 1)
			top=dets[0].top();
			bottom=shape.parts()[8].y;	#chin
			left=(dets[0].left()+dets[0].right())/2-(bottom-top)/2;
			right=(dets[0].left()+dets[0].right())/2+(bottom-top)/2;
			if bottom-top != right-left :
				left=left-1;
			if left<0 or right>=img.shape[0] :
				os.remove(f)
				print("Deleted")
			else :
				ff = open(pathname+'/'+outputname+".txt","w")
				for p in enumerate(shape.parts()):
					ff.write(str(p[1].x-left))
					ff.write(" ")
					ff.write(str(p[1].y-top))
					ff.write('\n')
				region = img[top:bottom,left:right]
				cv2.imwrite(f,region)
                detectedNum = detectedNum + 1
    else :
		os.remove(f)
		print("Deleted")
    if detectedNum == 10:
        os.remove(f)
