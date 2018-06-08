from data_reader import *
from PIL import Image
from statics import *

def resize(files,save_dir):

    for idx,path in enumerate(files):
        if 'Positive' in path:
            class_dir="Positive"
        elif 'Negative' in path:
            class_dir="Negative"
        else:
            class_dir="Neutral"

        save_class_dir=os.path.join(save_dir, class_dir)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)

        img=Image.open(path)
        img_np_de=np.asarray(img)
        if img_np_de.ndim!=3:
            print("{} {}".format(path,img_np_de.shape))
        continue

        save_file=os.path.join(save_class_dir,os.path.basename(path).split(".")[0]+".jpg")

        if img.layers==1:
            img_np=imageio.imread(path)
            img_np=img_np.reshape((img_np.shape[0],img_np.shape[1],1))
            #print(img_np.shape)
            img_np=np.repeat(img_np,3,2)
            #print(img_np.shape)
            imageio.imwrite(save_file, img_np)
            img = Image.open(save_file)
            img = img.resize((256, 256))
            img.save(save_file)
            continue
        img=img.resize((256,256))
        img.save(save_file)
        if idx%500==0:
            print(idx)

def resize_train():
    files = glob.glob('./Train/**/**')
    save_dir = './Train_256'
    resize(files,save_dir)

def resize_valid():
    files = glob.glob('./Validation_256/**/**')
    save_dir = './Validation_256'
    resize(files,save_dir)


if __name__ == '__main__':
    resize_valid()
