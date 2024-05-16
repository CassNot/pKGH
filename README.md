# PatchKGH (pKGH)
:recycle:*Garbage in, garbage out* - classic saying for data curation in Computer Vision  
The KinGston General (KGH) dataset for colorectal polyps classification consists of **1037 Whole Slide Images** (WSIs) encompassing both healthy colon tissue and colon polyps. The pathological WSIs have been annotated at the region of interest (ROI) level by Dr. Sonal Varma using a software developed by Huron Digital Pathology. In order to train deep learning models for colorectal polyp classification applied to this dataset, we need to extrat patches from the WSIs. To achieve a good representation of this dataset, we need to perform strong data curation and extract patches, or tiles, which are cleaned and do not present with any artefacts.
## :open_file_folder: Presentation of the dataset and its annotations
KGH dataset requires 1.2 TB of storage. All slides are stored as .tif files presenting with 4 resolution levels: 20X, 5X, 1.25X and 0.3125X. The figure below shows the different downsampled levels as well as a ROI annotation. The resolution at 20X is of 0.4 mpp (microns per pixels). The tissue thickness in these slides is of 5 microns.  
<div align="center">
  <img width="40%" alt="Design of all laboratory assignments" src="figures/pyramid_TVA75.png">
</div>
The polyps studied in this dataset are Sessile Serrated Lesions (SSL), Hyperplastic Polyps (HP), Tubulovillous Adenoma (TVA) and Tubular Adenoma (TA). This dataset is also presenting normal, or histological, colon tissues. One WSI can present multiple ROIs. The number of WSIs and annotations per class is given in the table below:

| Class  | Number of WSIs | Number of ROIs |
| ------------- | ------------- | ------------- |
| Normal (histology)  | 200  | 0  |
| Hyperplastic Polyps (HP) | 212 | 284 |
| Sessile Serrated Lesions (SSL) | 201 | 548 |
| Tubular Adenoma (TA) | 207 | 465 |
| Tubulovillous Adenoma (TVA) | 217 | 842 |

### Studied colorectal polyps
Four different types of colorectal polyps are studied in this dataset:
- **Sessile Serrated Lesions (SSL)**: SSLs, a subtype of serrated polyps, constitute at least 20% of all serrated polyps. They are considered precancerous and are characterized by a flat or slightly elevated shape, predominantly found in the cecum and ascending colon. Distinguishing SSLs from hyperplastic polyps may pose challenges, although certain distortions can be recognized by pathologists;
- **Hyperplastic Polyps (HP)**: These noncancerous growths carry a low risk of malignant transformation and are characterized by an overgrowth of normal cells in the mucosal lining of the colon or distal colon;
- **Tubular Adenoma (TA)**: These are usually small and benign polyps, prevalent in more than 80% of cases. While they are considered precancerous, less than 10% of them have the potential to progress into cancer;
-  **Tubulovillous Adenomas (TVA)**: TVAs are a subtype of colonic adenomas exhibiting a combination of tubular and villous features. Considered precancerous, they have the potential to transform into malignant structures. 
The figure below highlights the four different polyps and a normal patch, under a Field of View (FoV) of 800x800 microns
<div align="center">
  <img width="80%" alt="Design of all laboratory assignments" src="figures/5ex_800.png">
</div>

### Regions of interest (ROI) annotations
A pathologist annotated some regions of interest in the slide using an annotation tool from [Huron Digital Pathology](https://www.hurondigitalpathology.com). It is important to highlight that regions outside ROIs cannot be considered as normal tissue regions nor as pathological regions. They contain normal tissue, distorted tissue and pathological tissue which have not been annotated. The figure below shows 5 WSIs from the 5 different classes with ROI annotations and patches extracted from a ROI when applicable.
<div align="center">
  <img width="70%" alt="Design of all laboratory assignments" src="figures/ROIs_slides_patch.png">
</div>

## :person_fencing: Patch extraction challenges
## :mag: Patch extraction code
You will find the code for patch extraction in this [folder](patch-extraction).  
To run an extraction, you can precise '--datafolder' where the WSIs are stored, the Field of View ('--fov') of extraction, the level of extraction ('--level' between 0 and 2), the output folder for your new dataset ('--output') and the '--size' under which the patches will be saved:
```shell
python3 extraction_huron.py --size 256 --fov 800 --level 1 --method overlap --output ./kgh_800
```
A basic '--threshold 50' for patch/ROI overlap is fixed '--method overlap' is chosen (default). By default, the background and noisy patches are not saved, you can activate the saving using '--save True' and these patches will be saved in specific background folders.  
A [script](patch-extraction/extraction_script_example.sh) is given as an example.
## :computer: Conda environment for patch extraction
To work in the environment supporting [TIAToolBox](https://github.com/TissueImageAnalytics/tiatoolbox) and PyTorch (2.2.1), you can download the conda configuration file and follow the instructions below:
```shell
conda env create -f env.yml
```
Activate the environment:
```shell
conda activate tia-env
```
## :mailbox_with_mail: Contact information
For any questions, you can create an issue or send your question at [my courriel adress](mailto:cassandre.notton@mail.concordia.ca)
