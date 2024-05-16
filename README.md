# PatchKGH (pKGH)
:recycle:*Garbage in, garbage out* - classic saying for data curation in Computer Vision  
The KinGston General (KGH) dataset for colorectal polyps classification consists of **1037 Whole Slide Images** (WSIs) encompassing both healthy colon tissue and colon polyps. The pathological WSIs have been annotated at the region of interest (ROI) level by Dr. Sonal Varma using a software developed by Huron Digital Pathology. In order to train deep learning models for colorectal polyp classification applied to this dataset, we need to extrat patches from the WSIs. To achieve a good representation of this dataset, we need to perform strong data curation and extract patches, or tiles, which are cleaned and do not present with any artefacts.
## :open_file_folder: Presentation of the dataset and its annotations
KGH dataset requires 1.2 TB of storage. All slides are stored as .tif files presenting with 4 resolution levels: 20X, 5X, 1.25X and 0.3125X. The figure below shows the different downsampled levels as well as a ROI annotation. The resolution at 20X is of 0.4 mpp (microns per pixels). The tissue thickness in these slides is of 5 microns.  
The polyps studied in this dataset are Sessile Serrated Lesions (SSL), Hyperplastic Polyps (HP), Tubulovillous Adenoma (TVA) and Tubular Adenoma (TA). This dataset is also presenting normal, or histological, colon tissues. One WSI can present multiple ROIs. The number of WSIs and annotations per class is given in the table below:
| Class  | Number of WSIs | Number of ROIs |
| ------------- | ------------- | ------------- |
| Normal (histology)  | 200  | 0  |
| Hyperplastic Polyps (HP) | 212 | 284 |
| Sessile Serrated Lesions (SSL) | 201 | 548 |
| Tubular Adenoma (TA) | 207 | 465 |
| Tubulovillous Adenoma (TVA) | 217 | 842 |
## :person_fencing: Patch extraction challenges
## :mag: Patch extraction code
## Conda environment for patch extraction
## Contact information
