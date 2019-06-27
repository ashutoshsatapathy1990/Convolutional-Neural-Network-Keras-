from __future__ import print_function
import numpy as np
import cv2
import generator
import csv

#############################DATASET AND LABEL CREATION###############################

def extract():

	train_label_path = "train/label.csv"
	test_label_path = "test/label.csv"

	train_label = generator.gen(train_label_path)
	test_label = generator.gen(test_label_path)
	train_data = generator.image(train_label_path)
	test_data = generator.image(test_label_path)

	return (train_data, train_label, test_data, test_label)

#####################################END###############################################
