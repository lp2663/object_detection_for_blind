#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 06:37:56 2022

@author: levpaciorkowski

@IMPORTANT NOTE: This script should be executed two separate times to handle training and validation data

USAGE: python data_processing_yolov5.py labels.txt detections.csv labels_destination images_source images_destination

    labels.txt: a text file where each row is a label ID (corresponding to "LabelName" in Google Open Images V6)
        to be classified (have to write this text file yourself manually, picking the class labels you want to use)
        
    detections.csv: a csv file holding the bounding box detection labels (in Google Open Images V6 format)
    
    labels_destination: a directory where the individual image label .txt files should be written to
    
    images_source: the directory where all source images are located
    
    images_destination: a directory where the actual image files corresponding to the used labels should be copied to

"""

import pandas as pd
import sys
import shutil
pd.options.mode.chained_assignment = None


def create_label_translator(filename):
    label_translator = dict()
    file = open(filename, 'r')
    i = 0
    for line in file:
        label_translator[line.strip()] = i
        i += 1
    return label_translator

def generate_label(df_slice):
    yolo_label = ''
    image_id = list(df_slice['ImageID'])[0]
    objects = [label_translator[i] for i in list(df_slice['LabelName'])]
    x_centers = list(df_slice['x_center'])
    y_centers = list(df_slice['y_center'])
    widths = list(df_slice['width'])
    heights = list(df_slice['height'])
    for i in range(len(objects)):
        yolo_label += str(objects[i]) + ' ' + str(x_centers[i]) + ' ' + str(y_centers[i]) + ' ' + \
            str(widths[i]) + ' ' + str(heights[i])
        if i != len(objects) - 1: yolo_label += '\n'
    path = labels_destination + '/' + image_id + '.txt'
    with open(path, 'w') as f:
        f.write(yolo_label)
    image_from = images_source + '/' + image_id + '.jpg'
    image_to = images_destination + '/' + image_id + '.jpg'
    shutil.copyfile(image_from, image_to)
    return

def main(argv):
    # Load the data from provided arguments
    global label_translator
    label_translator = create_label_translator(argv[1])
    print("Reading detections labels in Google format...")
    detections = pd.read_csv(argv[2])
    global labels_destination
    labels_destination = str(argv[3])
    global images_source
    images_source = str(argv[4])
    global images_destination
    images_destination = str(argv[5])
    
    print("Extracting only our classes...")
    our_detections = detections.loc[detections['LabelName'].isin(label_translator.keys())] # very efficient operation
    
    # Translate from Google bounding box convention to YOLOv5
    our_detections['x_center'] = (our_detections['XMin'] + our_detections['XMax']) / 2
    our_detections['y_center'] = (our_detections['YMin'] + our_detections['YMax']) / 2
    our_detections['width'] = our_detections['XMax'] - our_detections['XMin']
    our_detections['height'] = our_detections['YMax'] - our_detections['YMin']
    
    # Now we need to write a label text file to disk for each image ID in our_detections
    l = our_detections['ImageID'].nunique()
    print('Writing ' + str(l) + ' Yolo label files to disk and copying selected images...')
    our_detections.groupby('ImageID').apply(generate_label)
    

if __name__ == "__main__":
    main(sys.argv)

