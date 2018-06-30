import os
import io
import requests
from PIL import Image
import tempfile
from utils import *


def download_image(img_url,save_file,size=(256, 256),total=None, progress=None):
    buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        downloaded = 0
        filesize = int(r.headers['content-length'])
        for chunk in r.iter_content():
            downloaded += len(chunk)
            buffer.write(chunk)
            if total==None:
                msg="Downloading: {}".format(save_file)
            else:
                msg="Downloading: {}, {} in {}".format(save_file,progress, total)

            progress_bar( int(downloaded*100/filesize),100, msg)
        buffer.seek(0)
        img = Image.open(io.BytesIO(buffer.read()))
        img = img.resize(size)
        img.save(save_file, quality=100)
    buffer.close()