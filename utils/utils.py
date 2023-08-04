import numpy as np
from PIL import Image
import os

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_img_student(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil=image_pil.resize((200, 250), Image.BICUBIC)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_img_test(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_numpy=image_numpy[8:248,8:248,:]
    image_pil = Image.fromarray(image_numpy)
    
    # image_pil=image_pil.resize((200, 250), Image.BICUBIC)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def zero_padding(img, size0, pad1, pad2):
    zero_padding = np.zeros((size0, size0), dtype=np.float32)
    pad1 = pad1 // 2
    pad2 = pad2 //2
    zero_padding[pad1:size0 - pad1, pad2:size0 - pad2] = img
    return zero_padding


def save_img_fusion_test(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    # print(image_numpy.shape)
    image_numpy=image_numpy[:,:,0]
    # image_numpy=image_numpy[8:248,8:248]
    image_numpy=zero_padding(image_numpy,240,16,16)
    image_pil = Image.fromarray(image_numpy).convert('L')

    # image_pil=image_pil.resize((240, 240), Image.BICUBIC)

    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def save_img_fusion(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    # print(image_numpy.shape)
    image_numpy=image_numpy[:,:,0]
    image_numpy=image_numpy[8:248,8:248]
    image_pil = Image.fromarray(image_numpy).convert('L')

    # image_pil=image_pil.resize((240, 240), Image.BICUBIC)

    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_img_fusion_real(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = image_numpy.clip(0, 255)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    # print(image_numpy.shape)
    image_numpy=image_numpy[:,:,0]
    # image_numpy=image_numpy[8:248,8:248]
    image_pil = Image.fromarray(image_numpy).convert('L')

    image_pil=image_pil.resize((240, 240), Image.BICUBIC)

    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_img_fusion_old_3(image_tensor, filename,transform):
    # print(filename,image_tensor.shape)
    image_numpy = image_tensor.float().numpy()
    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # print(image_numpy.shape)
    image_numpy = image_numpy.astype(np.uint8)
    image_numpy1=np.zeros((256,256,3))
    image_numpy1[:,:,0]=image_numpy[:,:,0]
    image_numpy1[:,:,1]=image_numpy[:,:,0]
    image_numpy1[:,:,2]=image_numpy[:,:,0]
    # 
    #image_numpy=image_numpy[:,:,0]
    image_numpy1 = image_numpy1.astype(np.uint8)
    # print("image_numpy1",image_numpy1.shape)
    image_pil = Image.fromarray(image_numpy1)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))



def save_img_fusion_old(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    image_numpy = image_numpy.astype(np.uint8)
 
    image_numpy=image_numpy[:,:,0]
    image_pil = Image.fromarray(image_numpy).convert('L')
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

import torch

def save_img_fusion_new(image_tensor, filename,transform):
    # print("image_tensor",image_tensor.shape)

    
    image_tensor1=torch.zeros((1,3,256,256))
    image_tensor1[:,0,:,:]=image_tensor
    image_tensor1[:,1,:,:]=image_tensor
    image_tensor1[:,2,:,:]=image_tensor   
    # print("image_tensor1",image_tensor1.shape)
    image_numpy = image_tensor1.float().numpy()
    
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

    image_numpy = image_numpy.astype(np.uint8)
    # print("image_numpy",image_numpy.shape)
    image_numpy=image_numpy[0,:,:,:]
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_img_fusion_weight(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  * 255.0

    image_numpy = image_numpy.astype(np.uint8)
 
    image_numpy=image_numpy[:,:,0]
    image_pil = Image.fromarray(image_numpy).convert('L')
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_img_fusion_weight3(image_tensor, filename,transform):
    image_numpy = image_tensor.float().numpy()

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  * 255.0


    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms

# def transform_convert(img_tensor, transform):
#     """
#     param img_tensor: tensor
#     param transforms: torchvision.transforms
#     """
#     if 'Normalize' in str(transform):
#         normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
#         mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
#         std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
#         img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
        
#     img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
#     if 'ToTensor' in str(transform) or img_tensor.max() < 1:
#         img_tensor = img_tensor.detach().numpy()*255
    
#     if isinstance(img_tensor, torch.Tensor):
#     	img_tensor = img_tensor.numpy()
    
#     if img_tensor.shape[2] == 3:
#         img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
#     elif img_tensor.shape[2] == 1:
#         img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
#     else:
#         raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
#     return img
