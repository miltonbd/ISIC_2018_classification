This repository contains the code for ISIC 2018 Skin Lesion Classification Task:

https://challenge2018.isic-archive.com/task3/

Possible disease categories are:

    Melanoma (MEL)
    Melanocytic nevus (NV)
    Basal cell carcinoma (BCC)
    Actinic keratosis / Bowen’s disease (intraepithelial carcinoma) (AKIEC)
    Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis) (BKL)
    Dermatofibroma (DF)
    Vascular lesion (VASC)
    
The input images for training should in a folder containing class specific folder:
 MEL, NV, BCC, AKIEC, BKL,DF,VASC
 
 python3 data_process.py will create the class specific folder structure, downlaod additional images 
 from isic archive and merge them in a common class specific folder.
    
Data Collection: The task 3 has 10015 training images. We downloaded additional images from https://isic-archive.com/#images.
we made a script that will
<img src="task3.png" />
 