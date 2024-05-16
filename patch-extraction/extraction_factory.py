# useful importations
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
import time
from PIL import Image, ImageOps, ImageFilter
import random
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import random
from torch import nn
import csv
import argparse
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon, box
from collections import OrderedDict
import math

class CustomModel(nn.Module):
    def __init__(self, input_channels, output_channels, nb_classes):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        # Linear layer
        self.fc = nn.Linear(output_channels * 50 * 50, nb_classes)  

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)
        # Forward pass through the linear layer
        x = self.fc(x)
        return x


# function to get all coordinates of all ROIs of this slide in one list
def get_rois_from_slide(file):
    tree = ET.parse(file)
    root = tree.getroot()

    number_ROIs = len(root[0])
    print("This slide presents ",number_ROIs, "ROIs")

    list_rois = []
    for roi in range(number_ROIs):
        vertices = root[0][roi][1]
        #print(vertices)
        list_coord = []
        for v in vertices:
            coord_ROI = v.attrib
            #print(coord_ROI['X'],coord_ROI['Y'])
            list_coord.append((int(coord_ROI['X']),int(coord_ROI['Y'])))
        list_rois.append(list_coord)

    return list_rois


# test if a point is inside one of the ROIs
def is_point_inside_roi(point, polygon_coords):
    # Create a Shapely Point
    point = Point(point)

    # Create a Shapely Polygon
    nb_rois = len(polygon_coords)
    for roi in range(nb_rois):
        polygon = Polygon(polygon_coords[roi])
        if point.within(polygon):
            return (True,roi)
    return (False,0)


# get center of one patch
def create_gravity_center(coords):
    list_grav = []
    for patch in coords:
        grav = ((patch[2]+patch[0])/2,(patch[3]+patch[1])/2)
        list_grav.append(grav)
    return list_grav

# define square polygons out of the diagonal coordinates
def make_square(coords):
    square = box(coords[0],coords[1],coords[2],coords[3])
    return square

# measure overlap between ROI and coords
def overlap_patch_roi(sq, polygon_coords,threshold):
    # returns True and ROI id if regions of overlap is above threshold and False,0 otherwise
    nb_rois = len(polygon_coords)
    for roi in range(nb_rois):
        polygon = Polygon(polygon_coords[roi])
        intersec = sq.intersection(polygon)
        relative_overlap = intersec.area/sq.area
        if relative_overlap > threshold/100:
            return (True,roi)
    return (False,0)
    

def ids_overlap_enough(all_coords,polygon_coords,threshold):
    ids = []
    roi_ids = []
    nb_patches = len(all_coords)
    for k in range(nb_patches):
        square = make_square(all_coords[k])
        decision = overlap_patch_roi(square,polygon_coords,threshold)
        if decision[0]:
            ids.append(k)
            roi_ids.append(decision[1])
    print(len(ids)/nb_patches, " percents of patches are from the ROI")
    return ids, roi_ids

# return ids of patches inside the ROI as well as the id of the ROI they belong to
def ids_to_keep(grav,rois):
    ids = []
    roi_ids = []
    nb_patches = len(grav)
    for k in range(nb_patches):
        result = is_point_inside_roi(grav[k], rois)
        if result[0]:
            ids.append(k)
            #print(grav[k])
            roi_ids.append(result[1])
    print(len(ids)/nb_patches, " percents of patches are from the ROI")
    return ids, roi_ids


def select_slides(x,total):
    selected_slides = []
    while len(selected_slides)<x:
        id = random.randint(0, total-1)
        if id not in selected_slides:
            selected_slides.append(id)
    return(selected_slides)

def coord_as_strings(coord):
    scoord = ''
    for k in range(4):
        scoord = scoord + '-'+str(int(coord[k]))
    return scoord