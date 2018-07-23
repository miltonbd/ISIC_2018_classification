from imgaug import augmenters as iaa
import imageio
from utils.utils_all import *
import time
import threading
from data_reader import *
"""
5Dtensor
todo save some random tensor.
"""

aug_save_dir="/media/milton/ssd1/research/competitions/EmotiW_2018/Train_aug"

def augment_images(a,b,c):
    pass

create_dir_if_not_exists(aug_save_dir)
def augment(images):
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Flipud(0.5),  # horizontal flips
            iaa.CropAndPad(percent=(-0.15, 0.15)),  # random crops
            iaa.Add((-40, 40)),
            iaa.GaussianBlur(sigma=(0, 0.5)),
            # Invert each image's chanell with 5% probability.
            # This sets each pixel value v to 255-v.
            iaa.Invert(0.05, per_channel=True),  # invert color channels

            # Add a value of -10 to 10 to each pixel.
            iaa.Add((-10, 10), per_channel=0.5),
        ]),

        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,

                      # Change brightness of images (50-150% of original value).
                      # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                      # iaa.ContrastNormalization((0.75, 1.5)),

                      # Improve or worsen the contrast of images.
                      # Convert each image to grayscale and then overlay the
                      # result with the original with random alpha. I.e. remove
                      # colors with varying strengths.
                      iaa.Affine(
                          scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                          translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                          rotate=(-10, 10),
                          shear=(-4, 4)
                      ),
                      iaa.Multiply((0.5, 1.5)),
                      iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                      iaa.Sequential([
                          iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                          iaa.WithChannels(0, iaa.Add((50, 100))),
                          iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
                      ]),
                      iaa.Superpixels(n_segments=100),
                      iaa.Invert(0.2),
                      iaa.CoarseSaltAndPepper(size_percent=0.05),
                      iaa.ElasticTransformation(2),
                      iaa.SimplexNoiseAlpha(
                          first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)),
                      iaa.FrequencyNoiseAlpha(
                          first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
                      ),
                      iaa.Grayscale(alpha=(0.0, 1.0)),
                      ),
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


    ], random_order=True)  # apply augmenters in random order

    seq_det = seq.to_deterministic()  # Call this once PER BATCH, otherwise you will always get the to get random
    # for i,img in enumerate(images):
    #     images_data[i,:,:,:]=img[:,:,:]
    aug_images = seq_det.augment_images(images)
    for i, aug_bb in enumerate(aug_images):
        idx_i = batch_idx + i
        imgid = train_dataset[idx_i]
        save_augs(JPEG_dir, anno_dir, idx_i, aug_images[i], aug_bb, imgid + "_" + str(idx_i))

def image_aug():
    pass
    # train_dataset = get_train_data()
    # train_dataset_aug=np.repeat(train_dataset,5,axis=0)
    # train_data_set = DatasetReader(train_dataset_aug, "train")
    # trainloader = torch.utils.data.DataLoader(train_data_set, batch_size=50,shuffle=True)
    # threads=[]
    # for batch_idx, (data, targets) in enumerate(trainloader):
    #     print(data.size())
    #     # save_augs(JPEG_dir,anno_dir,idx_i,aug_images,aug_bb)
    #     t = threading.Thread(target=augment, args=(data[0], data[1], batch_idx))
    #     threads.append(t)
    #     t.start()
    #     time.sleep(.01)
    #
    # for t in threads:
    #     t.join()

# image_aug()