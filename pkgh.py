import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import os
import pickle
from tqdm import tqdm


# for the patches dataset covering the whole KGH WSIs
# 15/04/2024 : support for kgh-1-25X

class pkgh(Dataset):
    def __init__(self, root_dir, size = 256, split = 'train', ROI = False, transform=None, balance = True):
        print("initialization of pkgh")
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            ])

        if transform:
            self.transform = transform

        self.data = []
        self.balance = balance

        pathologies = ['CP_TA','CP_TVA','CP_HP','CP_SSL','Normal']
    
        if split == 'train':
            # dataset under root / train / patho / ROI U nonROI / img
            dataset = os.path.join(self.root_dir,'train')
            if self.balance: 
                #only ROI
                ROI = True
                dataset = os.path.join(self.root_dir,"train_balanced_3500")
            for patho in pathologies:
                if patho != 'Normal':
                    # extracting all patches ROI + nonROI                              
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'ROI'))
                    images = [os.path.join("ROI",img) for img in all_images if img[-3:]=='png']

                    # adding images not from ROI as well
                    if not ROI:
                        all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'nonROI'))
                        images= images + [os.path.join("nonROI",img) for img in all_images if img[-3:]=='png']
                else:
                    # all normal patches are saved under root / train / Normal /
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho))
                    images = [img for img in all_images if img[-3:]=='png']  

                # assigning labels
                for image in images:
                    path_to_image = os.path.join(self.root_dir,dataset,patho,image)

                    class_name = 'N'
                    # assign label
                    if os.path.basename(image)[:2] == 'TA':
                        class_name = 'TA'
                    elif os.path.basename(image)[:2] == 'HP':
                        class_name = 'HP'
                    elif os.path.basename(image)[:3] == 'TVA':
                        class_name = 'TVA'
                    elif os.path.basename(image)[:3] == 'SSL':
                        class_name = 'SSL'
                    self.data.append([path_to_image, class_name])  



        if split == 'test':
            # dataset under root / test / patho / ROI U nonROI / img
            dataset = os.path.join(self.root_dir,'test')

            for patho in pathologies:

                if patho != 'Normal':
                    # extracting all patches ROI + nonROI                              
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'ROI'))
                    images = [os.path.join("ROI",img) for img in all_images if img[-3:]=='png']    
                    print(f"{len(images)} patches from ROI for train for {patho}") 

                    # adding images not from ROI as well
                    if not ROI:
                        all_images = os.listdir(os.path.join(self.root_dir,dataset,patho,'nonROI'))
                        images= images + [os.path.join("nonROI",img) for img in all_images if img[-3:]=='png']
                        print(f"{len(images)} patches from ROI and non ROI for test for {patho}") 
                else:
                    # all normal patches are saved under root / test / Normal /
                    all_images = os.listdir(os.path.join(self.root_dir,dataset,patho))
                    images = [img for img in all_images if img[-3:]=='png']    
                    print(f"{len(images)} patches from ROI and non ROI for test for {patho}") 

                # assigning labels
                for image in images:
                    path_to_image = os.path.join(self.root_dir,dataset,patho,image)

                    class_name = 'N'
                    # assign label
                    if os.path.basename(image)[:2] == 'TA':
                        class_name = 'TA'
                    elif os.path.basename(image)[:2] == 'HP':
                        class_name = 'HP'
                    elif os.path.basename(image)[:3] == 'TVA':
                        class_name = 'TVA'
                    elif os.path.basename(image)[:3] == 'SSL':
                        class_name = 'SSL'
                    self.data.append([path_to_image, class_name])
                
                       

        print("We have", len(self.data)," in this dataset")
        self.class_map = {"N" : 0, "TA": 1, "TVA": 2, "HP": 3, "SSL": 4}
        self.img_dim = (size, size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        #print(f"PATH = {img_path}")
        #print(f"--CLASS NAME = {class_name}")
        img = Image.open(img_path)
        img = self.transform(img)

        class_id = torch.tensor([self.class_map[class_name]])        
        return img, class_id

