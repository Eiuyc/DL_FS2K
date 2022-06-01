from pathlib import Path
import cv2, json, numpy as np
from PIL import Image
from tqdm import tqdm

import torch
torch.cuda.is_available()
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 定义Dataset
class DS(Dataset):
    def __init__(s, dataD, mode='train'):
        super().__init__()
        s.dataD = dataD
        s.mode = mode
        s.xtf = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
        ])
        s.ytf = transforms.Compose([
            torch.tensor,
        ])
        s.data = s.read()

    def read(s):
        D = s.dataD
        jp = D / f'anno_{s.mode}.json'
        with jp.open('r', encoding='utf-8') as f:
            annos = json.load(f)
        return annos

    def __getitem__(s, i):
        a = s.data[i]
        imgP = s.dataD / f"photo/{a['image_name']}.jpg"
        img = s.xtf(Image.open(imgP.as_posix()))
        colors = a['lip_color'] + a['eye_color']
        attrs = list(map(int, [a['hair'], a['hair_color'], a['gender'], a['earring'], a['smile'], a['frontal_face']]))
        return img, s.ytf(colors), torch.tensor(attrs, dtype=int)

    def __len__(s):
        return len(s.data)


# 实例化Dataset
dataD = Path('./FS2K')
train_ds = DS(dataD)
val_ds = DS(dataD, 'test')

# 创建Dataloader
train_dl = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=12, shuffle=True, num_workers=0)
