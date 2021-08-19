import torch
import PIL
from PIL import Image
import numpy as np
import os
import sys
import imageio
import argparse
import math
import random
import skimage
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rotate
from skimage.transform import rescale
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BCV_Dataset(torch.utils.data.Dataset):
    def __init__(self,datset_path,window_size):
      ''' Get the 3D image names.
          window_size : the crop size for the patch
          datset_path: data directory saved the label and training 3D image
      '''
      self.train_imgs_names = []
      self.label_imgs_names = []
      self.window_size = window_size
      self.datset_path = datset_path
      ###### Get the 3D image names
      for cur_dir in os.listdir(os.path.join(self.datset_path,'img/')):
          dir_img  = os.path.join(self.datset_path, 'label/', cur_dir)
          if os.path.isdir(dir_img):
              self.train_imgs_names.append(cur_dir)
              self.label_imgs_names.append(cur_dir)
      #print('train_imgs_names',self.train_imgs_names)
      #print('label_imgs_names',self.label_imgs_names)
      self.len = len(self.label_imgs_names)
      print('Number of testing images', self.len)
      self.to_tensor = transforms.Compose([
          transforms.ToTensor()
          ])
    def __getitem__(self,index):
      image_path = self.train_imgs_names[index]
      label_path = self.label_imgs_names[index]
      #print('image_path',image_path)
      #print('label_path',label_path)
      ###### read the 2D slices of 1 3D image into 3D numpy.ndarray
      image = self.imread3D(os.path.join(self.datset_path, 'img/', image_path))
      #print('image shape',image.shape)
      ###### read the 2D slices of 1 3D label into 3D numpy.ndarray
      label = self.imread3D_lbl(os.path.join(self.datset_path, 'label/', label_path))
      #print('label shape',label.shape)
      label = np.squeeze(label,axis=(0,))
      #print('label after squeeze shape',label.shape)
      ###### apply augmentation and random crop window size patch from image and label
      image_patch,label_patch = self.get_image(image,label,self.window_size)
      #print('image_patch shape',image_patch.shape)
      #print('label_patch shape',label_patch.shape)
      ###### convert the image_patch from numpy.ndarray to torch.FloadTensor
      image_patch = torch.from_numpy(image_patch).float()
      ###### convert the label_patch from numpy.ndarray to Tensor
      label_patch = torch.from_numpy(label_patch)
      return image_patch,label_patch
        
    def __len__(self):
      return self.len

    def imread3D(self,d):
      imgs = []
      ext = [".png", ".jpg"]
      for file in os.listdir(d):
        if file.endswith(tuple(ext)):
          imgs.append(file)
      imgs.sort()
      tempI = imread(os.path.join(d,imgs[0]))
      z = len(imgs)
      h = tempI.shape[0]
      w = tempI.shape[1]
      c = 1
      if (len(tempI.shape)>2):
        c = tempI.shape[2]
      #print(d)
      #print(c,z,h,w)
      I = np.zeros((c,z,h,w))
      idx = 0
      for i_name in imgs:
        tempI = skimage.img_as_float(imread(os.path.join(d,i_name)))
        if (len(tempI.shape)==2):
          I[0,idx,:,:] = tempI
        else:
          I[0,idx,:,:] = tempI[:,:,0]
          I[1,idx,:,:] = tempI[:,:,1]
          I[2,idx,:,:] = tempI[:,:,2]
        idx+=1
      '''nmax = np.percentile(I,99)
      nmin = np.percentile(I,1)
      print('enhance: ',nmax,nmin)
      I[np.greater(I,nmin)]=nmin
      I[np.less(I,nmax)] = nmax
      I = I-nmin
      I = I/(nmax-nmin)'''
      meanI = np.mean(I)
      stdI = np.std(I)
      I = (I-meanI)/stdI
      #print('norm:',I.max(),I.min(),np.mean(I),np.std(I))
      return I

    def imread3D_lbl(self,d):
      imgs = []
      ext = [".png", ".jpg"]
      for file in os.listdir(d):
        if file.endswith(tuple(ext)):
          imgs.append(file)
      imgs.sort()
      tempI = imread(os.path.join(d,imgs[0]))
      z = len(imgs)
      h = tempI.shape[0]
      w = tempI.shape[1]
      c = 1
      if (len(tempI.shape)>2):
        c = tempI.shape[2]
      #print(d)
      #print(c,z,h,w)
      I = np.zeros((c,z,h,w))
      idx = 0
      for i_name in imgs:
        tempI = skimage.img_as_float(imread(os.path.join(d,i_name)))
        if (len(tempI.shape)==2):
          I[0,idx,:,:] = tempI
        else:
          I[0,idx,:,:] = tempI[:,:,0]
          I[1,idx,:,:] = tempI[:,:,1]
          I[2,idx,:,:] = tempI[:,:,2]
        idx+=1
      #print(I.max())
      return I

#     def netpad(self,I,ws,pad):
#       #print(I.shape)
#       od = I.shape[1]
#       oh = I.shape[2]
#       ow = I.shape[3]
#       pd = max(od,ws) - od
#       ph = max(oh,ws) - oh
#       pw = max(ow,ws) - ow
#       hpd = math.floor(pd/2)
#       hph = math.floor(ph/2)
#       hpw = math.floor(pw/2)
#       if len(I.shape)==4:
#         imageType = I.shape[0] # C*DHW
#         nI0 = np.zeros((imageType,max(od,ws),max(oh,ws),max(ow,ws)))
#         nI0[:,hpd:hpd+od,hph:hph+oh,hpw:hpw+ow] = I
#         nI = np.zeros((imageType,nI0.shape[1]+2*pad,nI0.shape[2]+2*pad,nI0.shape[3]+2*pad))
#         nI[:,pad:pad+nI0.shape[1],pad:pad+nI0.shape[2],pad:pad+nI0.shape[3]] = nI0
#         return nI
#       nI0 = np.zeros((max(od,ws),max(oh,ws),max(ow,ws)))
#       nI0[hpd:hpd+od,hph:hph+oh,hpw:hpw+ow] = I
#       nI = np.zeros((nI0.shape[1]+2*pad,nI0.shape[2]+2*pad,nI0.shape[3]+2*pad))
#       nI[pad:pad+nI0.shape[1],pad:pad+nI0.shape[2],pad:pad+nI0.shape[3]] = nI0
#       return nI

    def get_image(self,train_img,label_img,ws):
      big_tI = train_img
      big_lI = label_img
      #print('big_tI shape',big_tI.shape)
      #print('big_lI shape',big_lI.shape)
      zs = big_tI.shape[1]
      xs = big_tI.shape[2]
      ys = big_tI.shape[3]
      imageType = big_tI.shape[0]
      stz = random.randint(0,zs-ws)
      stx = random.randint(0,xs-ws)
      sty = random.randint(0,ys-ws)
      train_sample = np.zeros((imageType,ws,ws,ws))
      label_sample = np.zeros((ws,ws,ws))
      train_sample = big_tI[:,stz:stz+ws,stx:stx+ws,sty:sty+ws]
      label_sample = big_lI[stz:stz+ws,stx:stx+ws,sty:sty+ws]
      flip = np.random.random()
      if flip>=0.5:
        train_sample = np.flip(train_sample, 3)
        label_sample = np.flip(label_sample, 2)
      nrotate = np.random.randint(0,3)
      train_sample = np.rot90(train_sample, nrotate, axes=(2,3))
      label_sample = np.rot90(label_sample, nrotate, axes=(1,2))
      label_sample = np.round(label_sample*255).astype('uint8')/255
      gstd = np.random.uniform(low=1/1.2,high=1.2)
      gmean = np.random.uniform(low=-0.2,high=0.2)
      train_sample = train_sample * gstd
      train_sample = train_sample + gmean
      return train_sample, label_sample

def data_main():
  ###### pytorch default: N*C*DHW
  datset_path = '/Users/gu/Desktop/PROJECTS/SSL_Pretraining_Projet/datasets/BCV/RawData/Training/'
  window_size = 64
  data = BCV_Dataset(datset_path,window_size)
  train_loader = DataLoader(dataset=data, 
                              num_workers=2,
                              shuffle=True, 
                              batch_size=7)
  print('number of batches', len(train_loader))
  for step, (images, labels) in enumerate(train_loader):
            print('test image shape',images.shape)
            print('test label shape',labels.shape)

  

if __name__ == '__main__':
  data_main()