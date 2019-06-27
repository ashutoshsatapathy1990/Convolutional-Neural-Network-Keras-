from __future__ import print_function
import numpy as np
import cv2
import csv
import os

##############################LABEL CREATION##################################

def gen(path):

	P = []

	with open(path) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			P.append(int(row[1]))

		P = np.array(P)

	return P

##################################END##########################################

############################DATASET CREATION###################################

def image(path):

	img, img1 = None, None
	with open(path) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			if img is None:
				img = cv2.imread(row[0])
				img = cv2.resize(img, (48, 48))
				img = img.reshape(1, 48, 48, 3)
				img1 = img
				
			else:
				img = cv2.imread(row[0])
				img = cv2.resize(img, (48, 48))
				img = img.reshape(1, 48, 48, 3)
				img1 = np.vstack([img1, img])
				
	return img1	

#####################################END########################################		
