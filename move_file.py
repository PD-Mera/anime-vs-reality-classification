from PIL import Image
from os import listdir
from os.path import join
from tqdm import tqdm
import random

ROOTPATH = '/home/teamai/TeamAI/dongtrinh/Anime-vs-Reality-Classification/data/full/Reality'
TARGETPATH_1 = '/home/teamai/TeamAI/dongtrinh/Anime-vs-Reality-Classification/data/valid/Reality'
TARGETPATH_2 = '/home/teamai/TeamAI/dongtrinh/Anime-vs-Reality-Classification/data/train/Reality'

list_dir = listdir(ROOTPATH)
random.shuffle(list_dir)
stop = len(list_dir) // 10

for idx, filename in tqdm(enumerate(list_dir)):
    fullname = join(ROOTPATH, filename)
    savename_1 = join(TARGETPATH_1, filename)
    savename_2 = join(TARGETPATH_2, filename)

    if idx < stop:
        if fullname.endswith(".jpg") or fullname.endswith(".png"):
            img = Image.open(fullname).resize((224, 224))
            img.save(savename_1)
    else:
        if fullname.endswith(".jpg") or fullname.endswith(".png"):
            img = Image.open(fullname).resize((224, 224))
            img.save(savename_2)
