from data_reader import *
from PIL import Image
from statics import *


def read_train_data_from_csv():
    csvfile=os.path.join(data_dir,"ISIC2018_Task3_Training_GroundTruth.csv")
    images=[]
    i = 0
    data=[]
    with open(csvfile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for line in csvreader:
            if i==0:
                print(line)
            else:
                img_path=os.path.join(data_dir,'ISIC2018_Task3_Training_Input',line[0]+".jpg")
                if os.path.exists(img_path):
                    images.append(img_path)
                    data.append([img_path,  np.argmax(line[1:]) ])
                else:
                    print("{} does not exists".format(img_path))
                    exit(0)
            i+=1
    return data


def resize(files,save_dir):
    files=np.asarray(files)
    images=files[:,0]
    labels=files[:,1]
    for idx, path in enumerate(images):
        class_dir=class_names[int(labels[idx])]
        save_class_dir=os.path.join(save_dir, class_dir)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)
        img=Image.open(path)
        img_np_de=np.asarray(img)
        if img_np_de.ndim!=3:
            print("{} {}".format(path,img_np_de.shape))
            continue
        # print(save_class_dir)
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

def train_test_split():
    data = read_train_data_from_csv()
    save_dir = os.path.join(data_dir,"Train_256")
    train_end_count = int(len(data) * .7)
    train_data = data[0:train_end_count]
    valid_data = data[train_end_count:]

    resize(train_data,save_dir)

    save_dir = os.path.join(data_dir,"Validation_256")

    resize(valid_data,save_dir)


if __name__ == '__main__':
    train_test_split()
    # resize_valid()
