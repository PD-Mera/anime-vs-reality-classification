# NSFW Anime Arts Classification

This project simply classifies whether image is **Anime** or **Reality**

## Environments

- Python 3.10.9
- torch 1.13.1
- torchvision 0.14.1

Install requirements

``` bash
pip install -r requirements.txt
```

## Data

- I get my data from various resource on internet. You can use your own data or crawl from internet.

- Data in this format

``` files
|-- data
    |-- train
    |   |-- class 1
    |   |-- class 2
    |   `-- ...
    `-- valid
        |-- class 1
        |-- class 2
        `-- ...
```

### Config

Modify config in `./cfg/config.yaml` or create your own `.yaml` config file with the same format.

### Train

Simply run 

``` bash
python train.py --cfg ./cfg/config.yaml
```

### Experiment Results

Some experiment results

| Model | Accuracy | Confusion Matrix | Pretrained | Model size |
| --- | :---: | :---: | :---: | :---: |
| **SqueezeNet1.1** | 99.34% | ![CM1](./assets/squeezenet1_1_confusion_matrix.jpg "CM1 Image") | [Model](https://drive.google.com/file/d/1NRX1JZ5thrajb5fugC1-dpp3rnGkoYUc/view?usp=share_link) | 4.74MB |
| **EfficientNet V2 Small** | 99.71% | ![CM2](./assets/efficientnetv2s_confusion_matrix.jpg "CM2 Image") | [Model](https://drive.google.com/file/d/1SqP6tmmuu3dirtxTtiVhKSVpTXMxpkpK/view?usp=share_link) | 82.74MB |

You can download weight file above and put in `weights` folder and run inference

``` bash
python infer.py
```

#### Some inference results

| Anime | Reality | 
| :---: | :---: |
| ![Anime](./assets/anime.jpg "Anime Image") | ![Reality](./assets/reality.png "Reality Image") |

You can try on your own :wink: