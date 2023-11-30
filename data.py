import zipfile
import shutil
import os

# os.makedirs(os.path.dirname('train/'))

with zipfile.ZipFile("dogs-vs-cats/train.zip", 'r') as zip_ref:
    zip_ref.extractall("train/")

os.makedirs(os.path.dirname('traindata/dog/'))
os.makedirs(os.path.dirname('traindata/cat/'))
os.makedirs(os.path.dirname('testdata/dog/'))
os.makedirs(os.path.dirname('testdata/cat/'))
train_dir = 'train/train/'
filenames = os.listdir(train_dir)
for file in filenames:
    if int(file[4:].replace(".jpg", "")) < 1000:
        if file[:3] == 'dog':
            shutil.copy(train_dir + file, 'testdata/dog/')
        else:
            shutil.copy(train_dir + file, 'testdata/cat/')
    else:
        if file[:3] == 'dog':
            shutil.copy(train_dir + file, 'traindata/dog/')
        else:
            shutil.copy(train_dir + file, 'traindata/cat/')

os.rmdir(os.path.dirname('train/'))