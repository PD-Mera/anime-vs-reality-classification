from os import listdir
from os.path import join
from torch.utils.data import Dataset
import random

from PIL import Image, ImageFilter
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomRotation
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def to_onehot_tensor(class_num, index, smooth = 0.9):
    label = [(1 - smooth) / (class_num - 1) for _ in range(class_num)]
    label[index] = smooth
    return torch.Tensor(label)


def random_gaussian_blur(img, thres = 0.5, radius_range = [2, 5]):
    rand = random.random()
    if rand < thres:
        radius = random.randint(radius_range[0], radius_range[1])
        img = img.filter(ImageFilter.GaussianBlur(radius = radius))
    return img


class LoadDataset(Dataset):
    def __init__(self, config: dict = None, phase = 'train'):
        """
            phase: 'train' or 'valid'
        """
        super(LoadDataset, self).__init__()
        self.config = config
        self.num_class = self.config['class']['num']
        self.images = []
        
        for classname in self.config['class']['name']:
            for filename in listdir(join(self.config[phase]['path'], classname)):
                if is_image_file(filename):
                    self.images.append(join(self.config[phase]['path'], classname, filename))


        self.transform = Compose([
            RandomRotation(30, expand=True),
            Resize(self.config['image_size']),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

  
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        # image = random_gaussian_blur(image, thres=1)
        image = self.transform(image)
        
        class_name = self.images[index].split('/')[-2]
        label = to_onehot_tensor(self.num_class, self.config['class']['name'].index(class_name), smooth=self.config['train']['smooth'])

        return image, label


    def __len__(self):
        return len(self.images)