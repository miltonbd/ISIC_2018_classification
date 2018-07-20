import os
import io
import requests
from PIL import Image
import tempfile
from utils import *
import threading
import time
thread_pools=[]
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # important to avoid

def download_images(img_urls,save_files,size=(256,256), no_threads=6):
    for i in range(len(img_urls)):
        while(True):
                img_url=img_urls[i]
                save_file=save_files[i]
                t=threading.Thread(target=download_image, args=(img_url,save_file,size,len(img_urls),(i+1)))
                if len(thread_pools)<no_threads:
                    t.start()
                    thread_pools.append(save_file)
                    break

import shutil
def download_image(img_url,save_file,size,total=None, progress=None):
    try:
        if os.path.exists(save_file):
            img=Image.open(save_file)
            if img is not None:
                thread_pools.remove(save_file)
                # shutil.copy(save_file,os.path.join('/media/milton/ssd1/research/competitions/ISIC_2018_data/data/aditional_training_data/MEL', os.path.basename(save_file)))
                return
    except Exception as e:
        pass
    progress_bar(progress,total,"Downloaded: {}".format(save_file))
    r = requests.get(img_url)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)
    with Image.open(io.BytesIO(r.content)) as img:
        img = img.convert('RGB')
        img.save(save_file, quality=100)
        img1 = Image.open(save_file)
        img1.resize(size)
        img1.save(save_file)
        thread_pools.remove(save_file)



