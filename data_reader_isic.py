from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
import torch
from torchvision import transforms
from statics_isic import *
import glob
from PIL import Image
import os
import threading

def padd_class(data, max):
    data=np.asarray(data)
    offset=max-data.shape[0]
    if offset!=0:
        last=data[-2:-1,:]
        offset_data=np.repeat(last,offset,0)
        new_data=np.concatenate((data,offset_data),axis=0)
    else:
        new_data=data
    return new_data


def get_data(files):
    data=[]
    mel=[]
    nv=[]
    bcc=[]
    akiec=[]
    bkl=[]
    df=[]
    vasc=[]
    for img in files:
        label=class_names.index(img.split('/')[-2])
        row=[img,int(label)]
        if int(label)==0:
            mel.append(row)
        elif int(label)==1:
            nv.append(row)
        elif int(label) == 2:
            bcc.append(row)
        elif int(label) == 3:
            akiec.append(row)
        elif int(label) == 4:
            bkl.append(row)
        elif int(label) == 5:
            df.append(row)
        elif int(label) == 6:
            vasc.append(row)
    counts=[len(mel),len(nv),len(bcc),len(akiec),len(bkl),len(df),len(vasc)]
    max_elements=max(counts)
    mel=padd_class(mel,max_elements)
    bcc=padd_class(bcc,max_elements)
    nv=padd_class(nv,max_elements)
    akiec=padd_class(akiec,max_elements)
    bkl=padd_class(bkl,max_elements)
    df=padd_class(df,max_elements)
    vasc=padd_class(vasc,max_elements)

    counts=[len(mel),len(nv),len(bcc),len(akiec),len(bkl),len(df),len(vasc)]
    print("mel: {}, nv: {}, bcc:{}, akiec:{},bkl:{},df:{}, vasc:{}".format(*counts))
    return np.concatenate([mel,nv,bcc,akiec,bkl,df,vasc],axis=0)
def get_train_data():
    files=glob.glob(os.path.join(data_dir,'Train_256/**/**.jpg'))
    # files=np.repeat(files,5,0)
    return get_data(files)

def get_validation_data():
    files=glob.glob(os.path.join(data_dir, 'Validation_256/**/**.jpg'))
    return get_data(files)

class DatasetReader(Dataset):
    """
    """
    def __init__(self, data,mode='train',):
        print("{} count:{}".format(mode,len(data)))
        self.mode=mode
        self.data=np.asarray(data)
        self.transform_train_image=transforms.Compose([
            transforms.RandomCrop([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]);
        self.transform_test_image = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()]);


    def __getitem__(self, index):
        img_path=self.data[index,0]
        label=int(self.data[index,1])

        if not os.path.exists(img_path):
            print("{} image not found".format(img_path))
            exit(0);
        img = Image.open(img_path)
        if self.mode=="train":
            data = self.transform_train_image(img)
            return data, label

        elif self.mode=="valid":
            data = self.transform_test_image(img)
            return data, label

    def __len__(self):
        return len(self.data)

def get_data_sets(batch_size1, batch_size2):
    train_data_set = DatasetReader(get_train_data(),"train")
    validation_data_set = DatasetReader(get_validation_data(),"valid")
    trainloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size1, shuffle=True,
                                              num_workers=2)
    valloader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size2, shuffle=False,
                                              num_workers=2)
    return (trainloader, valloader)

def test():
    trainloader, valloader = get_data_sets(100)
    for idx, (inputs, targets) in enumerate(valloader):
        print(inputs.shape)