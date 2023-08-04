from os import listdir
from os.path import join
import random
import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import SimpleITK as sitk
from utils import is_image_file, load_img
import scipy.ndimage as ndimage
import numpy as np

def zero_padding(img, size0, pad1, pad2):
    zero_padding = np.zeros((img.shape[0], size0, size0), dtype=np.float32)
    pad1 = pad1 // 2
    pad2 = pad2 //2
    zero_padding[:, pad1:size0 - pad1, pad2:size0 - pad2] = img
    return zero_padding

def resize(img, shape,order=1, mode='constant', orig_shape=(240, 240)):

    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1]
    )
    
    # Resize to the given shape
    return ndimage.zoom(img, factors, order=1,mode=mode)
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        
        self.t1_path = join(image_dir, "*/*t1.nii")
        self.t2_path = join(image_dir, "*/*t2.nii")
        self.flair_path = join(image_dir, "*/*flair.nii")


        self.image_filenames_t1= sorted(glob.glob(self.t1_path ))
        self.image_filenames_t2= sorted(glob.glob(self.t2_path ))
        self.image_filenames_flair = sorted(glob.glob(self.flair_path ))
        self.if_train=True
        if image_dir=='/home/guoliangqi/dataset/MICCAI_BraTS2020_ValidationData':
            self.if_train=False        



    def __getitem__(self, index):
        # a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        # b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        t1=sitk.GetArrayFromImage(sitk.ReadImage(self.image_filenames_t1[index]))
        t2=sitk.GetArrayFromImage(sitk.ReadImage(self.image_filenames_t2[index]))
        flair=sitk.GetArrayFromImage(sitk.ReadImage(self.image_filenames_flair[index]))

               
        # w_offset= random.randint(0, max(0, 286-256-1))
        # h_offset= random.randint(0, max(0, 286-256-1))
        #离群点检测 
        # np.percentile(0,99.5)

        num=78
        t1=t1[num,:,:]
        t1=torch.tensor(t1).type(torch.FloatTensor)     
        
           
        t1=resize(t1,(256, 256),order=1)
        t1=t1/t1.max() *2.0 -1 
        t1=torch.tensor(t1).type(torch.FloatTensor)
        t1 = t1.unsqueeze(0)   


        # input_t1 = input_t1.unsqueeze(0)
        t2=t2[num,:,:]            
        t2=torch.tensor(t2).type(torch.FloatTensor)    
           

        t2=resize(t2,(256, 256),order=1)    
        
        t2=t2/t2.max() *2.0 -1        
        t2=torch.tensor(t2).type(torch.FloatTensor)    
        t2 = t2.unsqueeze(0)

        flair=flair[num,:,:]
        flair=torch.tensor(flair).type(torch.FloatTensor)     
          
        
        flair=resize(flair,(256, 256),order=1)
        flair=flair/flair.max() *2.0 -1  

        flair=torch.tensor(flair).type(torch.FloatTensor) 
        flair = flair.unsqueeze(0)



        if self.if_train and random.random() < 0.5:
            idx = [i for i in range(t1.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            t1 = t1.index_select(2, idx)
            t2 = t2.index_select(2, idx)
            flair = flair.index_select(2, idx)
       
        return t1,t2,flair

    def __len__(self):
        return len(self.image_filenames_t1)
