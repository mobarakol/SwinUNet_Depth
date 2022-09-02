import os
import cv2
import argparse
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

    

class SimCol3DDataloader(Dataset):
    def __init__(self, subsets, subjects, transform=None, transform_target=None):
        self.transform = transform
        self.transform_target = transform_target
        self.img_dir_all = []
        for i, subset in enumerate(subsets):
            for subject in subjects[i]:
                subj_name = 'Frames_' + subject
                img_dir = glob(os.path.join(subset, subj_name, 'FrameBuffer**.png'))
                self.img_dir_all.extend(img_dir)
                
    def __len__(self):
        return len(self.img_dir_all)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir_all[idx]).convert('RGB')
        depth_dir = self.img_dir_all[idx][:-20] + 'Depth' + self.img_dir_all[idx][-9:]
        #depth = Image.open(depth_dir).convert('L')
        depth = cv2.imread(depth_dir, cv2.IMREAD_GRAYSCALE)
        depth = Image.fromarray(depth)
        img, depth = self.transform(img), self.transform_target(depth)

        return img, depth

def get_dataloader(args):
    train_subsets = ['../dataset/SyntheticColon_I_Train', 'dataset/SyntheticColon_II_Train']
    train_subjects = [['S1', 'S2','S3', 'S6', 'S7', 'S8', 'S11', 'S12', 'S13'], ['B1', 'B2','B3', 'B6', 'B7', 'B8', 'B11', 'B12', 'B13']]
    # val
    val_subsets = ['../dataset/SyntheticColon_I_Train', 'dataset/SyntheticColon_II_Train']
    val_subjects = [['S4', 'S9', 'S14'], ['B4', 'B9', 'B14']]
    if args.aug:
        transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ])

        transform_target = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    ])
        
    else:
        transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    transform_target = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                ])
    # train_dataset
    train_dataset = SimCol3DDataloader(train_subsets, train_subjects, transform=transform, transform_target=transform_target)
    trainloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=4)

    # Val_dataset
    val_dataset = SimCol3DDataloader(val_subsets, val_subjects, transform=transform, transform_target=transform_target)
    validloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size*2, shuffle=False, num_workers=4)
    print('Sample size- train:{}, valid:{}'.format(len(train_dataset), len(val_dataset)))
    return trainloader, validloader