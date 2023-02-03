import os
import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from model import init_model
from config import *
from utils import parse_opt, load_cfg


def draw_to_image(image: Image, text: str):
    draw = ImageDraw.Draw(image)
    h, w = image.size
    size = min(h, w) // 8
    font = ImageFont.truetype('assets/arial.ttf', size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font = font)
    draw.rectangle((left-5, top-5, right+5, bottom+5), fill="black")
    draw.text((0, 0), text, (255,255,255), font = font)
    return image


def test(config):
    os.makedirs(config['test']['result_path'], exist_ok=True)

    assert config['load_checkpoint'] is not None, "'load_checkpoint' in config must be specified"
    model = init_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    if os.path.isdir(config['test']['path']):  
        print(f"Starting infer {len(os.listdir(config['test']['path']))} images")
        for imagelink in tqdm(os.listdir(config['test']['path'])):
            if os.path.isdir(os.path.join(config['test']['path'], imagelink)):
                print(f"{os.path.join(config['test']['path'], imagelink)} is a folder. Exiting..")
                exit()
            image = Image.open(os.path.join(config['test']['path'], imagelink))
            h, w = image.size
            if h < 128 or w < 128:
                image = image.resize((h * 4, w * 4))
            inputs = transform(image).unsqueeze(0).to(device)
            output = model(inputs).cpu()
            results = torch.argmax(output)
            prob = output[0][results]
            classname = config['class']['name'][results]
            image = draw_to_image(image, classname + " - " + str(prob))
            image.save(os.path.join(config['test']['result_path'], imagelink))
        print(f"Image saved to {config['test']['result_path']}")
         
    elif os.path.isfile(config['test']['path']):  
        print(f"Starting infer 1 images")
        image = Image.open(config['test']['path'])
        h, w = image.size
        if h < 128 or w < 128:
            image = image.resize((h * 4, w * 4))
        inputs = transform(image).unsqueeze(0).to(device)
        output = model(inputs).cpu()
        results = torch.argmax(output)
        prob = float(output[0][results])
        classname = config['class']['name'][results]
        image = draw_to_image(image, f"{classname} - {prob * 100:.2f}%")
        image.save(os.path.join(config['test']['result_path'], config['test']['path'].split('/')[-1]))
        print(f"Image saved to {os.path.join(config['test']['result_path'], config['test']['path'].split('/')[-1])}")

    elif not os.path.exists(config['test']['path']):
        print("Infer path doesn't exist")
        exit()
    else:  
        print("Save path invalid")
        exit()

if __name__ == '__main__':
    opt = parse_opt()
    cfg = load_cfg(opt.cfg)
    test(config = cfg)