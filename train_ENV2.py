import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import dataLoader
import time
import effnetv2
import os
from torch.autograd import Variable
import tqdm

use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lr = 0.002
momentum = 0.9
num_epochs = 60


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label2):
        hair, hair_color, gender, earring, smile, frontal_face = pred
        # print(label2[:,0])

        # ll = self.l2(lip,label1[:, :3])
        # el = self.l2(eye,label1[:, 3:])
        # print(label2[:])
        hl = self.ce(hair, label2[:, 0])
        hcl = self.ce(hair_color, label2[:, 1])
        gl = self.ce(gender, label2[:, 2])
        eal = self.ce(earring, label2[:, 3])
        sl = self.ce(smile, label2[:, 4])
        fl = self.ce(frontal_face, label2[:, 5])
        loss = hl + hcl + gl + eal + sl + fl
        # print('ll:', ll.item(), 'el:', el.item(), 'hl:', hl.item(), 'hcl:', hcl.item(), 'gl:', gl.item(), 'eal:', eal.item(), 'sl:', sl.item(), 'fl:', fl.item())
        return loss


def exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=80):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def acc_cal(prama, model, val_dl, use_gpu=False):
    vn = len(val_dl.dataset)
    model.load_state_dict(prama)
    hair_cnt = 0
    hair_color_cnt = 0
    gender_cnt = 0
    earring_cnt = 0
    smile_cnt = 0
    frontal_face_cnt = 0
    with torch.no_grad():
        for data in val_dl:
            inputs, label1, label2 = data
            if use_gpu:
                inputs, label1, label2 = Variable(inputs.cuda()), Variable(label1.cuda()), Variable(label2.cuda())
            else:
                inputs, label1, label2 = Variable(inputs), Variable(label1), Variable(label2)
            output = model(inputs)
            hair, hair_color, gender, earring, smile, frontal_face = output

            hair_ = torch.max(hair, 1)[1]
            hair_cnt += torch.sum(hair_ == label2[:, 0])

            hair_color_ = torch.max(hair_color, 1)[1]
            hair_color_cnt += torch.sum(hair_color_ == label2[:, 1])

            gender_ = torch.max(gender, 1)[1]
            gender_cnt += torch.sum(gender_ == label2[:, 2])

            earring_ = torch.max(earring, 1)[1]
            earring_cnt += torch.sum(earring_ == label2[:, 3])

            smile_ = torch.max(smile, 1)[1]
            smile_cnt += torch.sum(smile_ == label2[:, 4])

            frontal_face_ = torch.max(frontal_face, 1)[1]
            frontal_face_cnt += torch.sum(frontal_face_ == label2[:, 5])
    avgAcc = (hair_cnt + hair_color_cnt + gender_cnt + earring_cnt + smile_cnt + frontal_face_cnt) / 6 / vn
    print(f'validated on {vn} samples| mean acc:{avgAcc * 100:.4f}%')
    print(f'hair_cnt:{hair_cnt / vn} | hair_color_cnt:{hair_color_cnt / vn} | gender_cnt:{gender_cnt / vn}')
    print(f'earring_cnt:{earring_cnt / vn} | smile_cnt:{smile_cnt / vn} | frontal_face_cnt:{frontal_face_cnt / vn}')

    return avgAcc


def train_model(model_ft, optimizer, lr_scheduler, num_epochs=2000, use_gpu=True):
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)
    model_eval = model_ft.eval()

    dset_loaders = dataLoader.train_dl
    dset_sizes = dset_loaders.batch_size
    dset_val = dataLoader.val_dl

    for epoch in tqdm.trange(num_epochs):

        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        for data in dset_loaders:
            inputs, label1, label2 = data
            if use_gpu:
                inputs, label1, label2 = Variable(inputs.cuda()), Variable(label1.cuda()), Variable(label2.cuda())
            else:
                inputs, label1, label2 = Variable(inputs), Variable(label1), Variable(label1)

            outputs = model_ft(inputs)
            # print(outputs)
            l = Loss()
            loss = l(outputs, label2)
            # _, preds = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 30 == 0 or outputs[0].shape[0] < dset_sizes:
                # print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())

            running_loss += loss.item()

        epoch_acc = acc_cal(model.state_dict(), model_eval, dset_val, use_gpu=True)
        epoch_loss = running_loss / dset_sizes

        print(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} ')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()
        if epoch_acc > 0.999:
            break

    # save best model
    save_dir = 'save/model'
    os.makedirs(save_dir, exist_ok=True)
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + 'EN' + '.pth'
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss, best_model_wts


if __name__ == '__main__':
    model = effnetv2.effnetv2_s()
    model = model.cuda()

    optimizer = optim.SGD((model.parameters()), lr=lr,
                          momentum=momentum, weight_decay=0.0004)
    losses, prama = train_model(model_ft=model, optimizer=optimizer, lr_scheduler=exp_lr_scheduler, use_gpu=use_gpu)
    with open('./losee.txt', 'w') as f:
        for l in losses:
            f.write(l)
