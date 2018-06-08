import os
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np

def augment_images(sources_list, multiply_augmented):
    source_images = []
    original_images = []
    for source_img_path in sources_list:
        img = Image.open(source_img_path)
        crop_rectangle = (13, 13, 224, 224)
        cropped_im = img.crop(crop_rectangle)
        resized = cropped_im.resize((112, 112), Image.ANTIALIAS)
        source_images.append(np.asarray(resized))
        original_image=np.asarray(resized).copy()
        original_images.append(original_image)
    source_images = np.asarray(source_images)
    original_images =np.asarray(original_images)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.05)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),

        # Strengthen or weaken the contrast in each image.
        # iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # rotate=(-25, 25),
            shear=(-4, 4)
        ),
        iaa.Grayscale(alpha=(0.0, 1.0))
    ], random_order=True)  # apply augmenters in random order

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    images = np.asarray(source_images).astype(np.float32)
    images = np.repeat(images, multiply_augmented, 0)
    # print(images.shape)
    images_aug = seq.augment_images(images)
    images_aug_arr = np.asarray(images_aug)
    images_aug_arr = np.concatenate((original_images, images_aug_arr), 0)
    # print(images_aug_arr.shape)
    return  images_aug_arr
