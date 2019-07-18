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
# from progressbar import *
# from focal_loss import FocalLoss
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
# import layer
import cv2
import data_argmentation as DA

from densenet import densenet121, densenet161,densenet201,densenet169
import pretrainedmodels
from grad_loss import Grad_loss,pertubation,cal_max_grad
import logging
# import visdom
# import torch.distributed as dist

model_class_map = {
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet201': densenet201,
    'densenet169': densenet169,
}

classify_map = {}
class_map = {}
for line in open("classify.csv","r"):
    line = line.strip('\n')
    key = line.split(',')[0].zfill(5)
    value = line.split(',')[1]
    classify_map[key] = value

for line in open("class3.csv","r"):
    line = line.strip('\n')
    key = line.split(',')[0].zfill(5)
    value = line.split(',')[1]
    class_map[key] = value

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
        image_aug = image_aug.astype(np.float32)
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
    all_labels = [int(class_map[img_path.split('/')[-2]]) for img_path in all_imgs]

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
                       num_workers=8,
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
    image_quan = transforms.Lambda(lambda images : (images*100).type(torch.cuda.IntTensor).type(torch.cuda.FloatTensor)/100)
    # focal_loss = FocalLoss(class_num=num_classes)
    # focal_loss = focal_loss.cuda()
    # grad_loss = Grad_loss().cuda()
    best_acc = 0.0
    # trainloss_vis = visdom.Visdom(env='train_loss')
    time_step = 0
    mean_train_loss = []
    # do training
    for i_ep in range(n_ep):
        model.train()
        train_losses = []
        # widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
        #            ' ', ETA(), ' ', FileTransferSpeed()]
        # pbar = ProgressBar(widgets=widgets)
        count = 0
        for batch_data in train_loader:
            image = batch_data['image'].type(torch.FloatTensor).to(device)
            label = batch_data['label_idx'].to(device)
            x = Variable(image,requires_grad = True)

            if np.random.randint(2):

                optimizer_mini = torch.optim.SGD([x], weight_decay=1e-5, lr=100)
                for i in range(1):
                    model.eval()
                    logits = model(x)
                    loss = -criterion(logits, label)
                    loss.backward(retain_graph=True)
                    max_grad = cal_max_grad(x)
                    optimizer_mini.step()
                    torch.clamp(image, min=-1.0, max=1.0, out=image)
                    if float(max_grad[0]) >= 0.5:
                        break
            image = image_quan(image)
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            # logits = Classifier(logits,label)
            # loss = focal_loss(logits, label)#F.cross_entropy
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_losses += [loss.detach().cpu().numpy().reshape(-1)]
            count += 1
            if count % 40 ==0:
                mean_train_loss.append(np.concatenate(train_losses).mean())
                time_step += 1
                # trainloss_vis.line(Y=torch.tensor(mean_train_loss),X=torch.range(1,time_step),win=line)
        train_losses = np.concatenate(train_losses).reshape(-1).mean()

        model.eval()
        val_losses = []
        preds = []
        true_labels = []
        # widgets = ['val:', Percentage(), ' ', Bar('#'), ' ', Timer(),
        #            ' ', ETA(), ' ', FileTransferSpeed()]
        # pbar = ProgressBar(widgets=widgets)
        for batch_data in val_loader:
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
        logging.debug(
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

    # pretrained_model = models.densenet121(pretrained = True)
    # pretrained_dict = pretrained_model.state_dict()

    num_classes = len(class_map)
    # Model = model_class_map[model_name]
    # model = Model(num_classes=num_classes)

    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # model = torch.load('./save_weights/new_experiment/grad_loss/ep_23_densenet121_val_acc_0.9123.pkl')

    model = pretrainedmodels.pnasnet5large(num_classes=1000)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    # pretrained = torch.load('./pytorch_weights/pnasnet5large/ep_14_pnasnet5large_val_acc_0.7167.pkl')
    # pretained_dict = pretrained.module.state_dict()
    # model.load_state_dict(pretained_dict)
    model.train()

    # Loading data for ...
    logging.debug('loading data for train %s ....' % model_name)
    dataset_dir = '/home/wangzhaowei/rongyf/data/seven_classes/3'

    img_size = 331
    loaders = load_data_for_training_cnn(dataset_dir, img_size, batch_size=batch_size * len(gpu_ids))

    # Prepare training options
    save_path = './pytorch_weights/seven_class/%s' % model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.debug("Let's use ", len(gpu_ids), " GPUs!")
    device = torch.device('cuda:%d' % gpu_ids[0])
    model = model.to(device)

    # dist.init_process_group(init_method='file:/home/wangzhaowei/rongyf',backend='gloo',world_size=2)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        # model = torch.nn.parallel.DistributedDataParallel(model)
        

    logging.debug('start training cnn model.....\nit will take several hours, or even dozens of....')
    do_train(model_name, model, loaders['train_data'], loaders['val_data'],
             device, lr=0.0001, save_path=save_path, n_ep=30, num_classes=num_classes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', default='densenet121',
                        help='cnn model, e.g. , densenet121, densenet161', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=8,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename='r_pnas_class3.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    ######################################################

    args = parse_args()
    gpu_ids = args.gpu_id
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    batch_size = args.batch_size
    target_model = args.target_model
    train_cnn_model('pnasnet5large_class3', gpu_ids, batch_size)


    #######################################################
    
