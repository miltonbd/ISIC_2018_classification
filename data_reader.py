from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
import torch
from torchvision import transforms
from statics import *
import glob
from PIL import Image
import os

def get_data(files):
    images=[]
    labels=[]

    for img in files:
        images.append(img)
        if 'Positive' in img:
            labels.append(0)
        elif 'Negative' in img:
            labels.append(1)
        else:
            labels.append(2)

    print('images:{}'.format(len(images)))
    return (images,labels)
def get_train_data():
    files=glob.glob('./Train_256/**/**.jpg')
    return get_data(files)

def get_validation_data():
    files=glob.glob('./Validation_256/**/**.jpg')
    return get_data(files)

class DatasetReader(Dataset):
    """
    task1 =melanoma
    task2=sebrorreheic
    mode= train, validation, test
    """
    def __init__(self, images, labels,mode='train',):
        self.mode=mode
        self.images=images
        self.labels=labels
        self.transform_train_image=transforms.Compose([
            transforms.RandomCrop([224,224]),
            transforms.ToTensor()]);
        self.transform_test_image = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()]);


    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.mode=="train":
            if not os.path.exists(self.images[index]):
                print("{} image not found".format(self.images[index]))
                exit(0);

            data = self.transform_train_image(img)
            label = self.labels[index]
            return (data, label)

        elif self.mode=="valid":
            if not os.path.exists(self.images[index]):
                print("{} image not found".format(self.images[index]))
                exit(0);

            label = self.labels[index]
            data = self.transform_test_image(img)
            return (data, label)


    def __len__(self):
        return len(self.images)

def get_data_sets(model_details):
    augmentor=model_details.augmentor
    images, labels=get_train_data()
    train_data_set = DatasetReader(images, labels ,"train")
    validation_data_set = DatasetReader( *get_validation_data(),"valid")
    return (train_data_set, validation_data_set)

if __name__ == '__main__':
    train_images, train_labels=get_validation_data()
    trainset=DatasetReader(train_images, train_labels,"valid")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                                   num_workers=2)
    for idx,(images,labels) in enumerate(trainloader):
        print(images.shape)


