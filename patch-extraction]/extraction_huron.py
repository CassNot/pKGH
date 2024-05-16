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
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import random
import logging
import tiatoolbox
import csv
import argparse
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
from tiatoolbox import logger
from tiatoolbox.wsicore import wsireader
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.misc import imread, read_locations
from collections import OrderedDict
import math
from extraction_factory import select_slides, get_rois_from_slide, create_gravity_center, ids_to_keep, coord_as_strings, CustomModel, ids_overlap_enough
from torchvision.transforms.functional import rotate


parser = argparse.ArgumentParser(description='patch extraction')
parser.add_argument('--datafolder',default = '/home/huron/Documents/Datasets/KGH_WSIs')
parser.add_argument('--output',default = "/home/huron/Documents/Datasets/KGH_patches",type = str)
parser.add_argument('--fov',default=544,type=int)
parser.add_argument('--size',default=1024,type=int)
parser.add_argument('--overlap',default=10,type=int)
parser.add_argument('--save',default = False,type = bool)
parser.add_argument('--level',default = 1, choices = [0,1,2,3], type=int)
parser.add_argument('--method',default = 'center',type = str)
parser.add_argument('--threshold',default = 50, type = int)

args = parser.parse_args()


##### MODEL FOR NOISE/BCK REMOVAL #####

# transformation to be applied before forwarding the images
transform = transforms.Compose([
    transforms.Resize(50),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8808, 0.7753, 0.8206], std=[0.0853, 0.1287, 0.1086]),
    
])

# device selection
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# model definition
input_channels = 3  # Assuming RGB input
output_channels = 64  # Number of output channels for the final Conv2D layer
nb_classes = 2  # Size of the output for the linear layer
model = CustomModel(input_channels,output_channels,nb_classes)
model.to(device)
model.load_state_dict(torch.load("./best-model-Custom-test-128-act-20.ckpt"))
model.eval()
print("--- model loaded")


# for normal: extract all patches from the tissue
# for pathologies: extract all patches from the ROI in a folder and all patches not from ROI in another one
# if args.save : will save background patches in another folder

def main(args=args,model=model,transform=transform):


    args.bck_folder = os.path.join(args.output,"bck")

    ## initialization of the txt file to store the extraction summary (parameters and nb patches/slide)
    args.output_txt = os.path.join(args.output, "extraction_summary.txt")
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    with open(args.output_txt, "w") as file:
        file.write(f"Extracting from {args.datafolder} to {args.output}\n")
        file.write(f"FoV is {args.fov} and overlap is {args.overlap}\n")
        file.write(f"Extraction done at level {args.level}\n")
        file.write(f"Background patches saved: {args.save}\n")
        file.write(f"Threshold for overlap if {args.method} is {args.threshold}")


    RESHAPE_SIZE = args.size
    with open(args.output_txt, "a") as file:
        file.write(f"Patches will be stored at size {RESHAPE_SIZE}\n")
    print("Extracting patches from ",args.datafolder, "with FoV of ",args.fov, "micrometers and resized to ",RESHAPE_SIZE)

    ## specifying the name of folders for each class
    PATHOLOGIES = ["Normal","CP_HP","CP_TA","CP_TVA","CP_SSL"]
    splits = ['test','train']

    for split in splits:
        if not os.path.exists(os.path.join(args.output,split)):
            os.mkdir(os.path.join(args.output,split))

        for pathology in PATHOLOGIES:
        
            OUTPUT_FOLDER = os.path.join(args.output,split,pathology)
            
            if not os.path.exists(OUTPUT_FOLDER):
                os.mkdir(OUTPUT_FOLDER)

            if pathology == "Normal":
                # extraction from normal slides with small overlap
                args.overlap = 10
                extraction_from_normal(args,pathology,split)

            else:
                # extraction from pathological slides with bigger overlap
                print("--- extracting from ROI")
                args.overlap = 30

                print("Extracting ",pathology," with overlap of ",args.overlap)
                extraction_from_roi(args,pathology,split)
                

def get_prediction_score(model,pil_image,transform,RESHAPE_SIZE):
    # returns the prediction score of an image with the model (0 if tissue, 1 otherwise)
    with torch.no_grad():
        patchr = pil_image.resize((RESHAPE_SIZE,RESHAPE_SIZE))
        image = transform(patchr).unsqueeze(0).to(device)
        output = model(image)
        output = output.cpu()
        _,predicted = torch.max(output, 1)
        predicted = int(predicted)
    return predicted


##### PATCH EXTRACTION FROM PATHOLOGICAL SLIDES ##### 
def extraction_from_roi(args,pathology,split, transform=transform):
    #we extract at level 0 because annotations have been made and are more precised at level 0
    print("Extraction from ",pathology)
    path_to_pathology = os.path.join(args.datafolder,split,pathology)
    
    OUTPUT_FOLDER_ROI = os.path.join(args.output,split,pathology,"ROI")
    OUTPUT_FOLDER_nonROI = os.path.join(args.output,split,pathology,"nonROI")

    if not os.path.exists(OUTPUT_FOLDER_ROI):
        os.mkdir(OUTPUT_FOLDER_ROI)

    if not os.path.exists(OUTPUT_FOLDER_nonROI):
        os.mkdir(OUTPUT_FOLDER_nonROI)

    # extraction from the slides with annotations only
    xmls = [item for item in os.listdir(path_to_pathology) if item.endswith('.xml')]
    all_wsis = [xml[:-4]+'.tif' for xml in xmls] 

    print(all_wsis)
    # go through all WSIs in the folder
    for wsi in all_wsis:
        t_s = time.time()
        wsi_path = os.path.join(path_to_pathology,wsi)
        print("Extracting patches from ",wsi)
        patches_extracted = 0
        patches_extracted_ROI = 0

        #reading the WSI
        slide = WSIReader.open(wsi_path)
        INFO_WSI = slide.info.as_dict()
        SHAPE_WSI = INFO_WSI['level_dimensions'][0]
        print(INFO_WSI)
        res = slide.convert_resolution_units(input_res = 0, input_unit = 'level', output_unit = 'mpp')[0]
        PATCH_SIZE = int(args.fov/res)  # int(224*INFO_WSI['level_downsamples'][args.level])
        print(f"Patch size at this level = {PATCH_SIZE}")
        STRIDE = int((100-args.overlap)/100*PATCH_SIZE) 
        RESHAPE_SIZE = args.size

        # get coordinates from all ROIs
        roi_coords = get_rois_from_slide(os.path.join(path_to_pathology,wsi[:-4]+'.xml'))

        # computing mask from slide using TIAToolBox
        otsumask = slide.tissue_mask(method = 'otsu', resolution = 1, units = 'level')
        print("--- Otsumask done")

        # extracting patches from tissue regions
        patches_in_slide = patchextraction.get_patch_extractor(
            input_img = slide,
            input_mask = otsumask,
            method_name="slidingwindow",  # also supports "point" and "slidingwindow"
            patch_size=(
                PATCH_SIZE,
                PATCH_SIZE,
            ),  # size of the patch to extract around the centroids from centroids_list
            stride=(STRIDE, STRIDE),
            resolution=0,
            units="level",
        )
        coordinates = patches_in_slide.coordinate_list

        print('--- patches extracted using tiatoolbox')
        # adapting the coordinates to the referential of the ROIs
        h = INFO_WSI['slide_dimensions'][1]
        for k in range(len(coordinates)):
            coordinates[k][1],coordinates[k][3]=h-coordinates[k][1],h-coordinates[k][3]
        
        # test if center of patches inside ROI
        if args.method == 'center':
            # this methods tests if the center of the patch is within the ROI
            gravs = create_gravity_center(coordinates)
            to_keep = ids_to_keep(gravs, roi_coords)
        elif args.method == 'overlap': 
            # this methods tests if the overlap between the patch and the ROI is above a threshold (args.threshold)
            to_keep = ids_overlap_enough(coordinates,roi_coords,args.threshold)

        t = 0
        print("--- running second filter (CustomModel)")   
        if args.save:
            BCK_FOLDER = os.path.join(args.output,split,pathology,'bck')
            if not os.path.exists(BCK_FOLDER):
                os.mkdir(BCK_FOLDER)

        if len(to_keep[0])>0:
            for id in range(len(patches_in_slide)):

                # ROI patches
                if id in to_keep[0]:

                    #fixing id of the roi to which the patch belong
                    if len(roi_coords)==1:
                        id_roi = 0
                    else:
                        id_roi = to_keep[1][t]
                    t += 1

                    patch = patches_in_slide[id]

                    #second filter aligned with KGH dataset requirements
                    
                    pil_image = Image.fromarray(patch)                    
                    predicted = get_prediction_score(model,pil_image,transform,RESHAPE_SIZE)
                    
                    if predicted==0: #tissue
                        name_patch = wsi[:-4]+"-r"+str(id_roi)+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+".png"
                        patchr = pil_image.resize((RESHAPE_SIZE,RESHAPE_SIZE))
                        patchr.save(os.path.join(OUTPUT_FOLDER_ROI,name_patch))

                        #adding augmentations for HP patches
                        r = 90
                        if random.random()>0.5:
                            r = 270
                        if pathology == 'CP_HP':
                            name_patch = wsi[:-4]+"-r"+str(id_roi)+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+"-rot"+str(r)+".png"
                            rot_patch = rotate(patchr,90)
                            rot_patch.save(os.path.join(OUTPUT_FOLDER_ROI,name_patch))
                            patches_extracted_ROI += 1
                        patches_extracted_ROI += 1

                    # save bck patches if applicable
                    else:
                        if args.save:
                            pil_image = Image.fromarray(patch)
                            name_patch = wsi[:-4]+"-r"+str(id_roi)+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+".png"
                            patchr = pil_image.resize((50,50))
                            patchr.save(os.path.join(BCK_FOLDER,name_patch))
                
                # non ROI but tissue
                else:
                   
                    patch = patches_in_slide[id]

                    #second filter aligned with KGH dataset requirements
                    
                    pil_image = Image.fromarray(patch)                    
                    predicted = get_prediction_score(model,pil_image,transform,RESHAPE_SIZE)
                    
                    if predicted==0: #tissue
                        name_patch = wsi[:-4]+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+".png"
                        patchr = pil_image.resize((RESHAPE_SIZE,RESHAPE_SIZE))
                        patchr.save(os.path.join(OUTPUT_FOLDER_nonROI,name_patch))
                        
                        #adding augmentations for HP patches
                        if pathology == 'CP_HP':
                            name_patch = wsi[:-4]+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+"-rot90.png"
                            rot_patch = rotate(patchr,90)
                            rot_patch.save(os.path.join(OUTPUT_FOLDER_nonROI,name_patch))
                            name_patch = wsi[:-4]+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+"-rot270.png"
                            rot_patch = rotate(patchr,270)
                            rot_patch.save(os.path.join(OUTPUT_FOLDER_nonROI,name_patch))
                            patches_extracted += 2
                        patches_extracted += 1

                    # save bck patches if applicable
                    else:
                        if args.save:
                            pil_image = Image.fromarray(patch)
                            name_patch = wsi[:-4]+"-"+str(id)+'-'+coord_as_strings(coordinates[id])+".png"
                            patchr = pil_image.resize((50,50))
                            patchr.save(os.path.join(BCK_FOLDER,name_patch))


        print("---", patches_extracted, " patches have been saved")
        print("Saving patches from this slide took ",time.time()-t_s," seconds")
        with open(args.output_txt, "a") as file:
            file.write(f"{wsi[:-4]} | {str(patches_extracted)} | {str(patches_extracted_ROI)} \n")
        print("--- Updated output txt")
        print("Extraction from ",pathology," is done")





##### NORMAL PATCH EXTRACTION AT LEVEL ASKED ##### 
def extraction_from_normal(args,pathology,split, transform=transform):
    print("Extracting from the normal slides")
    path_to_pathology = os.path.join(args.datafolder,split,pathology)
    OUTPUT_FOLDER = os.path.join(args.output,split,pathology)
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
        
    all_wsis = [item for item in os.listdir(path_to_pathology) if item.endswith('.tif')]

    # go through all WSIs in the folder
    for wsi in all_wsis:
        t_s = time.time()
        wsi_path = os.path.join(path_to_pathology,wsi)
        print("Extracting patches from ",wsi)
        patches_extracted = 0

        #reading the WSI
        slide = WSIReader.open(wsi_path)
        INFO_WSI = slide.info.as_dict()
        print(INFO_WSI)

        #precising the size of patch to be extracted at the level asked in argument
        res = slide.convert_resolution_units(input_res = args.level, input_unit = 'level', output_unit = 'mpp')[0]
        PATCH_SIZE = int(args.fov/res)
        STRIDE = int((100-args.overlap)/100*PATCH_SIZE) 
        RESHAPE_SIZE = args.size
        if PATCH_SIZE<RESHAPE_SIZE: #we do not want upsampling
            print("Saving patches with size at extraction = ",PATCH_SIZE)
            RESHAPE_SIZE = PATCH_SIZE

        #masking the WSI to remove background using OtsuMask at level 2
        otsumask = slide.tissue_mask(method = 'otsu', resolution = 1, units = 'level')
        print("--- Otsumask done")
        #extracting patches from tissue regions
        tissue_patches = patchextraction.get_patch_extractor(
            input_img = slide,
            input_mask = otsumask,
            method_name="slidingwindow",  # also supports "point" and "slidingwindow"
            patch_size=(
                PATCH_SIZE,
                PATCH_SIZE,
            ),  # size of the patch to extract around the centroids from centroids_list
            stride=(STRIDE, STRIDE),
            resolution=args.level,
            units="level",
        )
        coordinates = tissue_patches.coordinate_list
        print('--- patches extracted using tiatoolbox')

        #second filter aligned with KGH dataset requirements
        print("--- running second filter (CustomModel)")
        if args.save:
            BCK_FOLDER = os.path.join(OUTPUT_FOLDER,'bck')
            if not os.path.exists(BCK_FOLDER):
                os.mkdir(BCK_FOLDER)

        for k in range(len(tissue_patches)):
            patch = tissue_patches[k]
            pil_image = Image.fromarray(patch)

            predicted = get_prediction_score(model,pil_image,transform,RESHAPE_SIZE)

                #save if tissue
            if predicted == 0:
                name_patch = wsi[:-4] + "-" + str(k)+"-"+coord_as_strings(coordinates[k])+".png"
                patchr = pil_image.resize((RESHAPE_SIZE,RESHAPE_SIZE))
                patchr.save(os.path.join(OUTPUT_FOLDER,name_patch))
                patches_extracted += 1
            else:
                if args.save:
                    name_patch = wsi[:-4] + "-" + str(k)+"-"+coord_as_strings(coordinates[k])+".png"
                    patchr = pil_image.resize((RESHAPE_SIZE,RESHAPE_SIZE))
                    patchr.save(os.path.join(BCK_FOLDER,name_patch))
                    patches_extracted += 1

        print("---", patches_extracted, " patches have been saved")
        print("Saving patches from this slide took ",time.time()-t_s," seconds")
        with open(args.output_txt, "a") as file:
            file.write(f"{wsi[:-4]} | {str(patches_extracted)}\n")
        print("--- Updated output txt")
    print("Extraction from Normal is done")




main()
