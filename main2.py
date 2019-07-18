import os
import PIL
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from progressbar import *
from focal_loss import FocalLoss
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import layer
import cv2
import data_argmentation as DA

from densenet import densenet121, densenet161,densenet201,densenet169

from grad_loss import Grad_loss,pertubation


model_class_map = {
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet201': densenet201,
    'densenet169': densenet169,
}


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer
        self.augment = DA.SSDAugmentation()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = cv2.imread(image_path)
        image_aug,__,___ = self.augment(image,None,image_path)
        image_tensor = self.transformer(image_aug)
        # image = self.transformer(Image.open(image_path))  # .convert('RGB'))
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image'      : image_tensor,
            'label_idx'  : label_idx,
            'filename'   : os.path.basename(image_path)
        }
        return sample


def load_data_for_training_cnn(dataset_dir, img_size, batch_size=16):
    all_imgs = glob.glob(os.path.join(dataset_dir, './*/*.jpg'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train,
                                            stratify=train['label_idx'].values, train_size=0.9, test_size=0.1)
    transformer_train = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transformer = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer),
        'val_data'  : ImageSet(val_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders


def load_data_for_defense(input_dir, img_size, batch_size=16):
    all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path': all_img_paths, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

def accuracy_score(true_labels,preds):
    return np.sum([1 for i in range(len(true_labels)) if(true_labels[i]==preds[i])])/len(true_labels)

def do_train(model_name, model, train_loader, val_loader, device, lr=0.0001, n_ep=40, num_classes=3,
             save_path='/tmp'):

    # Classifier = layer.MarginCosineProduct(1664, num_classes)
    # Classifier = Classifier.cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
    criterion = nn.CrossEntropyLoss().cuda()
    focal_loss = FocalLoss(class_num=num_classes)
    focal_loss = focal_loss.cuda()
    grad_loss = Grad_loss().cuda()
    best_acc = 0.0
    # do training
    for i_ep in range(n_ep):
        model.train()
        train_losses = []
        widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(train_loader):
            image = batch_data['image'].type(torch.FloatTensor).to(device)
            label = batch_data['label_idx'].to(device)
            x = Variable(image,requires_grad = True)
            optimizer.zero_grad()
            logits = model(x)
            # logits = Classifier(logits,label)
            # loss = focal_loss(logits, label)#F.cross_entropy
            loss = criterion(logits, label)
            loss.backward()
            if np.random.randint(2):
                pertubation(x,image)
                model.zero_grad()
                logits = model(x)
                loss = criterion(logits, label)
                # loss += loss2
                loss.backward()
            optimizer.step()
            train_losses += [loss.detach().cpu().numpy().reshape(-1)]
        train_losses = np.concatenate(train_losses).reshape(-1).mean()

        model.eval()
        val_losses = []
        preds = []
        true_labels = []
        widgets = ['val:', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(val_loader):
            image = batch_data['image'].type(torch.FloatTensor).to(device)
            label = batch_data['label_idx'].to(device)
            # image = Variable(image, requires_grad=True)
            with torch.no_grad():
                logits = model(image)
            # logits = Classifier(logits,label)
            loss = criterion(logits, label).detach().cpu().numpy().reshape(-1)
            # loss2 = grad_loss(image)
            # loss += loss2.detach().cpu().numpy().reshape(-1)
            # loss = focal_loss(logits, label).detach().cpu().numpy().reshape(-1)
            val_losses += [loss]
            true_labels += [label.detach().cpu().numpy()]
            preds += [(logits.max(1)[1].detach().cpu().numpy())]

        preds = np.concatenate(preds, 0).reshape(-1)
        true_labels = np.concatenate(true_labels, 0).reshape(-1)
        acc = accuracy_score(true_labels, preds)
        val_losses = np.concatenate(val_losses).reshape(-1).mean()
        scheduler.step(val_losses)
        # need python3.6
        print(
            f'Epoch : {i_ep}  val_acc : {acc:.5%} ||| train_loss : {train_losses:.5f}  val_loss : {val_losses:.5f}  |||')
        if acc > best_acc:
            best_acc = acc
            files2remove = glob.glob(os.path.join(save_path, 'ep_*'))
            for _i in files2remove:
                os.remove(_i)
            torch.save(model.cpu().state_dict(),
                       os.path.join(save_path, f'ep_{i_ep}_{model_name}_val_acc_{acc:.4f}.pth'))
            torch.save(model,os.path.join(save_path,f'ep_{i_ep}_{model_name}_val_acc_{acc:.4f}.pkl'))
            model.to(device)
        elif i_ep%10 == 9:
            torch.save(model.cpu().state_dict(),
                       os.path.join(save_path, f'ep_{i_ep}_{model_name}_val_acc_{acc:.4f}.pth'))
            torch.save(model, os.path.join(save_path, f'ep_{i_ep}_{model_name}_val_acc_{acc:.4f}.pkl'))
            model.to(device)


def train_cnn_model(model_name, gpu_ids, batch_size):
    # Define CNN model

    pretrained_model = models.densenet121(pretrained = True)
    pretrained_dict = pretrained_model.state_dict()

    num_classes = 3
    Model = model_class_map[model_name]
    model = Model(num_classes=num_classes)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model = torch.load('./save_weights/new_experiment/grad_loss/ep_23_densenet121_val_acc_0.9123.pkl')

    # Loading data for ...
    print('loading data for train %s ....' % model_name)
    dataset_dir = '/home/feng/Documents/class3_data'

    img_size = model.input_size[0]
    loaders = load_data_for_training_cnn(dataset_dir, img_size, batch_size=batch_size * len(gpu_ids))

    # Prepare training options
    save_path = './pytorch_weights/%s' % model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Let's use ", len(gpu_ids), " GPUs!")
    device = torch.device('cuda:%d' % gpu_ids[0])
    model = model.to(device)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    print('start training cnn model.....\nit will take several hours, or even dozens of....')
    do_train(model_name, model, loaders['train_data'], loaders['val_data'],
             device, lr=0.0001, save_path=save_path, n_ep=40, num_classes=num_classes)


def defense(input_dir, target_model, weights_path, defense_type, defense_params, output_file, batch_size):
    # Define CNN model
    Model = model_class_map[target_model]
    # defense_fn = defense_method_map[defense_type]
    model = Model(num_classes=110)
    # Loading data for ...
    print('loading data for defense using %s ....' % target_model)
    img_size = model.input_size[0]
    loaders = load_data_for_defense(input_dir, img_size, batch_size)

    # Prepare predict options
    device = torch.device('cuda:0')
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    pth_file = glob.glob(os.path.join(weights_path, 'ep_*.pth'))[0]
    print('loading weights from : ', pth_file)
    model.load_state_dict(torch.load(pth_file))

    # for store result
    result = {'filename': [], 'predict_label': []}
    # Begin predicting
    model.eval()
    widgets = ['dev_data :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(loaders['dev_data']):
        image = batch_data['image'].to(device)
        filename = batch_data['filename']
        with torch.no_grad():
            logits = model(image)
        y_pred = logits.max(1)[1].detach().cpu().numpy().tolist()
        result['filename'].extend(filename)
        result['predict_label'].extend(y_pred)
    print('write result file to : ', output_file)
    pd.DataFrame(result).to_csv(output_file, header=False, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', default='densenet121',
                        help='cnn model, e.g. , densenet121, densenet161', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=32,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gpu_ids = args.gpu_id
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    batch_size = args.batch_size
    target_model = args.target_model
    ################## Training #######################
    train_cnn_model(target_model, gpu_ids, batch_size)
    ################## Defense #######################
    # defense(input_dir, target_model, weights_path, defense_type, defense_params, output_file, batch_size)

