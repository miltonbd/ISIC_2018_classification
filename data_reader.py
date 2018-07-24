from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
import torch
from torchvision import transforms
from statics_isic import *
import glob
from PIL import Image
import os
from torchvision.transforms import *
import threading
num_classes=7
height=400
width=400
data_set_name="ISIC 2018"

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
    # max_elements=max(counts)
    # mel=padd_class(mel,max_elements)
    # bcc=padd_class(bcc,max_elements)
    # nv=padd_class(nv,max_elements)
    # akiec=padd_class(akiec,max_elements)
    # bkl=padd_class(bkl,max_elements)
    # df=padd_class(df,max_elements)
    # vasc=padd_class(vasc,max_elements)

    counts=[len(mel),len(nv),len(bcc),len(akiec),len(bkl),len(df),len(vasc)]
    print("mel: {}, nv: {}, bcc:{}, akiec:{},bkl:{},df:{}, vasc:{}".format(*counts))
    return np.concatenate([mel,nv,bcc,akiec,bkl,df,vasc],axis=0)
def get_train_data():
    files=glob.glob(os.path.join(data_dir,'Train_512_all/**/**.jpg'))
    data = []
    mel = []
    nv = []
    bcc = []
    akiec = []
    bkl = []
    df = []
    vasc = []
    for img in files:
        label = class_names.index(img.split('/')[-2])
        row = [img, int(label)]
        if int(label) == 0:
            mel.append(row)
        elif int(label) == 1:
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
    counts = [len(mel), len(nv), len(bcc), len(akiec), len(bkl), len(df), len(vasc)]
    # max_elements=max(counts)
    # mel=padd_class(mel,max_elements)
    # bcc=padd_class(bcc,max_elements)
    # nv=padd_class(nv,max_elements)
    # akiec=padd_class(akiec,max_elements)
    # bkl=padd_class(bkl,max_elements)
    # df=padd_class(df,max_elements)
    # vasc=padd_class(vasc,max_elements)
    counts = [len(mel), len(nv), len(bcc), len(akiec), len(bkl), len(df), len(vasc)]
    print("mel: {}, nv: {}, bcc:{}, akiec:{},bkl:{},df:{}, vasc:{}".format(*counts))
    data=np.concatenate([mel, nv, bcc, akiec, bkl, df, vasc], axis=0)
    np.random.shuffle(data)
    # data=np.repeat(data,2,axis=0)
    return data

def get_validation_data():
    files=glob.glob(os.path.join(data_dir, 'Validation_512/**/**.jpg'))
    return get_data(files)

class DatasetReader(Dataset):
    """
    """
    def __init__(self, images, mode='train', ):
        print("{} count:{}".format(mode, len(images)))
        self.mode=mode
        self.images=np.asarray(images)
        self.transform_train_image=transforms.Compose([
            transforms.CenterCrop(420),
            RandomCrop([400,400]),
            RandomHorizontalFlip(p=.2),
            # ColorJitter(.6),
            RandomVerticalFlip(p=.2),
            # RandomGrayscale(p=.2),
            transforms.RandomRotation(5),
            # transforms.RandomAffine(10),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]);

        self.transform_test_image = transforms.Compose([
            transforms.CenterCrop(420),
            RandomCrop([400, 400]),
            transforms.ToTensor()]);


    def __getitem__(self, index):
        try:
            img_path=self.images[index, 0]
        except Exception as e:
            img_path=self.images[index]

        if not os.path.exists(img_path):
            print("{} image not found".format(img_path))
            exit(0);
        img = Image.open(img_path)
        if self.mode=="train":
            data = self.transform_train_image(img)
            label = int(self.images[index, 1])
            return data, label

        elif self.mode=="valid":
            data = self.transform_test_image(img)
            label = int(self.images[index, 1])
            return data, label

        elif self.mode == "test":
            data = self.transform_test_image(img)
            return data

    def __len__(self):
        return len(self.images)
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def get_data_loader(batch_size):
    train_Data = get_train_data()
    train_data_set = DatasetReader(train_Data, "train")
    validation_data_set = DatasetReader(get_validation_data(),'valid')
    # weights = make_weights_for_balanced_classes(validation_data_set[0,:], len())
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,def get_data_loader(batch_size):22.07.2018)
    #
    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
    #                                            sampler=sampler, num_workers=args.workers, pin_memory=True)

    class_sample_count = np.repeat(0,num_classes)  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    train_Data = get_train_data()
    for train_data_row in train_Data:
        index = int(train_data_row[1])
        class_sample_count[index]=class_sample_count[index]+1
    class_sample_count=class_sample_count/len(train_Data)
    class_sample_count=1/class_sample_count

    weights = []
    for train_Data_row in train_Data:
        weight=class_sample_count[int(train_data_row[1])]
        weights.append(weight)
    weights=torch.Tensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    trainloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=sampler)


    valloader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size, shuffle=False,
                                              num_workers=2)
    return (trainloader, valloader)

def test():
    trainloader, valloader = get_data_loader(100)
    for idx, (inputs, targets) in enumerate(valloader):
        print(inputs.shape)


def get_validation_loader_for_upload(batch_size):
    test_files=glob.glob("/media/milton/ssd1/research/competitions/ISIC_2018_data/data/Validation_upload/**.jpg")
    test_data_set = DatasetReader(test_files,'test')
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=False,
                                              num_workers=2)
    return testloader

def get_test_loader_for_upload(batch_size):
    test_files=glob.glob("/media/milton/ssd1/research/competitions/ISIC_2018_data/data/Test_512/**.jpg")
    test_data_set = DatasetReader(test_files,'test')
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=False,
                                              num_workers=2)
    return testloader