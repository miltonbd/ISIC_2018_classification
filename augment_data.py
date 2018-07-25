from imgaug import augmenters as iaa
import imageio
from utils.utils_all import *
import time
import threading
import warnings
warnings.filterwarnings("ignore")
from data_reader import *
import warnings
warnings.filterwarnings("ignore")
import math
"""
todo save some random tensor.
"""

aug_save_dir="/media/milton/ssd1/research/competitions/ISIC_2018_data/data/Train_512_aug"

create_dir_if_not_exists(aug_save_dir)

def augment(images, labels, batch_idx):
    seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        # Strengthen or weaken the contrast in each image.
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, 0.5)),
            # Invert each image's chanell with 5% probability.
            # This sets each pixel value v to 255-v.
            # iaa.Invert(0.05, per_channel=True),  # invert color channels

            # Add a value of -10 to 10 to each pixel.
            iaa.Fliplr(0.3),  # horizontal flips
            iaa.Flipud(0.3),  # horizontal flips


            iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            shear=(-4, 4)
        ),
        #     iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.95, 1.1)),
        #
            # iaa.Superpixels(n_segments=10),
            # iaa.CoarseSaltAndPepper(size_percent=0.05),
            iaa.ElasticTransformation(2),
            # iaa.SimplexNoiseAlpha(
            #     first=iaa.Multiply(iaa.Choice([0.95, 1.05]), per_channel=True)
            # ),
        ]),

    ], random_order=True)  # apply augmenters in random order

    seq_det = seq.to_deterministic()  # Call this once PER BATCH, otherwise you will always get the to get random
    # for i,img in enumerate(images):
    #     images_data[i,:,:,:]=img[:,:,:]
    class_names_batch=[class_names[int(i)] for i in labels]
    # images=[]
    # for img_path in data:
    #     images.append(imageio.imread(img_path))
    images=images.transpose((0,2,3,1))
    aug_images = seq_det.augment_images(images)
    for i, aug_image in enumerate(aug_images):
        idx_i = batch_idx + i
        class_name=class_names_batch[i]
        class_dir=os.path.join(aug_save_dir,class_name)
        create_dir_if_not_exists(class_dir)
        save_path=os.path.join(class_dir,str(idx_i)+'.jpg')
        # print(save_path)
        imageio.imwrite(save_path,aug_image)
    if batch_idx%10==0:
        print("{} batch_idx done.".format(batch_idx))

def image_aug():
    batch_size=10
    train_dataset = get_train_data()
    train_dataset_aug=np.repeat(train_dataset,2,axis=0)
    train_data_set = DatasetReader(train_dataset_aug, "train")
    trainloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,shuffle=True)
    threads=[]
    iterations=len(trainloader)
    for batch_idx,(data,targets) in enumerate(trainloader):
        # min_size=batch_idx*batch_size
        # max_size=min(min_size+batch_size,len(train_dataset))
        # data=train_dataset[min_size:max_size,0]
        # targets=train_dataset[min_size:max_size,1]
        # print(data.size())
        # save_augs(JPEG_dir,anno_dir,idx_i,aug_images,aug_bb)
        # augment(data, targets, batch_idx)
        t = threading.Thread(target=augment, args=(data.cpu().data.numpy(), targets, batch_idx))
        threads.append(t)
        t.start()
        time.sleep(.01)
        # pass

    for t in threads:
        t.join()

image_aug()