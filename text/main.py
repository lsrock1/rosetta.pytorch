from model import build_model, loss_calculation, get_accuracy
from dataloader import build_data_loader
from transforms import build_transforms
import torch.optim as optim
import os
import time
import torch
import argparse
import cv2
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(cfg, model=None, to_file=False):
    if to_file and os.path.exists('./results.txt'):
        os.remove('./results.txt')
    transforms = build_transforms(is_train=False)
    data_loader = build_data_loader(cfg.dataset, 1, cfg.workers, transforms, is_train=False)
    if model is None:
        model = build_model(data_loader.dataset.num_of_classes)
        model = model.to(device)
        load_dict(cfg, model, cfg.load_name)
    model.eval()
    running_loss, running_corrects, running_all, running_all_correct = 0., 0., 0., 0.
    with torch.no_grad():
        
        running_loss, running_corrects, running_all, running_all_correct = 0., 0., 0., 0.
        for batch_idx, (data, targets, target_lengths) in enumerate(data_loader):
            
            data = data.to(device)
            target_lengths = target_lengths.to(device)
            predicted = model(data)
            loss = loss_calculation(predicted, torch.cat(targets, dim=0).to(device), target_lengths)
            
            running_loss += loss.item() * data.size(0)
            running_corrects += get_accuracy(predicted, targets, data_loader.dataset.ind_to_class, to_file)
            running_all_correct += torch.sum(target_lengths).item()
            #print('sum : ', running_all_correct)
            if to_file:
                # print(data.cpu().numpy().squeeze().transpose(1,2,0))
                # print(data.cpu()*255)
                # torchvision.utils.save_image(data.cpu()*255, './test/{}.jpg'.format(batch_idx))
                cv2.imwrite('./test/{}.jpg'.format(batch_idx), data.cpu().numpy().squeeze().transpose(1,2,0)*255)
            running_all += len(data)
            if batch_idx == 0:
                since = time.time()
            elif (batch_idx == len(data_loader)-1):#batch_idx % cfg.interval == 0 or 
                print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(data_loader.dataset),
                    100. * batch_idx / (len(data_loader)-1),
                    running_loss / running_all,
                    running_corrects / running_all_correct,
                    time.time()-since,
                    (time.time()-since)*(len(data_loader)-1) / batch_idx - (time.time()-since))),
    return running_corrects / running_all_correct

def train_model(cfg):
    transforms = build_transforms(is_train=True)
    data_loader = build_data_loader(cfg.dataset, cfg.batch_size, cfg.workers, transforms, is_train=True)
    model = build_model(data_loader.dataset.num_of_classes)
    model = model.to(device)
    model.train()
    optimizer = get_optimizer(cfg, model)
    if cfg.start_epoch > 0:
        load_dict(cfg, model, cfg.start_epoch-1, optimizer)
    best_accu = 0.
    for epoch in range(cfg.start_epoch, cfg.epochs):
        model.train()
        running_loss, running_corrects, running_all, running_all_correct = 0., 0., 0., 0.
        for batch_idx, (data, targets, target_lengths) in enumerate(data_loader):
            
            data = data.to(device)
            target_lengths = target_lengths.to(device)
            predicted = model(data)
            loss = loss_calculation(predicted, torch.cat(targets, dim=0).to(device), target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item() * data.size(0)
                running_corrects += get_accuracy(predicted, targets, data_loader.dataset.ind_to_class)
                running_all_correct += torch.sum(target_lengths).item()
                #print('sum : ', running_all_correct)
                running_all += len(data)
            if batch_idx == 0:
                since = time.time()
            elif batch_idx % cfg.interval == 0 or (batch_idx == len(data_loader)-1):
                print('Process: [epoch {} {:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    epoch,
                    running_all,
                    len(data_loader.dataset),
                    100. * batch_idx / (len(data_loader)-1),
                    running_loss / running_all,
                    running_corrects / running_all_correct,
                    time.time()-since,
                    (time.time()-since)*(len(data_loader)-1) / batch_idx - (time.time()-since))),
        val_accu = test_model(cfg, model)
        if val_accu > best_accu:
            best_accu = val_accu
            save_state(cfg, model, optimizer, 'best', best_accu)
    save_state(cfg, model, optimizer, cfg.epochs, val_accu)

def get_optimizer(cfg, model):
    return optim.SGD(model.parameters(), cfg.lr)

def save_state(cfg, model, optimizer, epoch, accuracy):
    if epoch == 'best':
        data = torch.load(os.path.join(cfg.resume_path, str(epoch)+'.pth'))
        if 'model' in data and data['accuracy'] > accuracy:
            return
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': accuracy,
        'epoch': epoch}, os.path.join(cfg.resume_path, str(epoch)+'.pth'))
    print('Model {}, Accuracy {}'.format(epoch, accuracy))

def load_dict(cfg, model, load_number, optimizer=None):
    data = torch.load(os.path.join(cfg.resume_path, str(load_number)+'.pth'))
    if 'model' in data:
        model.load_state_dict(data['model'])
        if optimizer is not None:
            optimizer.load_state_dict(data['optimizer'])
        print('Model loaded epoch {} accuracy {}'.format(data['epoch'], data['accuracy']))
    else:
        #구버전
        model.load_state_dict(data)
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(os.path.join(cfg.resume_path, str(load_number)+'_op.pth')))
# def get_scheduler(cfg, optimizer):
#     # optim.lr_scheduler.StepLR(optimizer, step_size=)

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def main():
    # Settings
    parser = argparse.ArgumentParser(description='rosetta resnet18 head')
    parser.add_argument('--dataset', default='.', help='path to dataset dir')
    parser.add_argument('--resume_path', default='./results', help='path to trained data')
    parser.add_argument('--lr', default=0.0003, help='initial learning rate', type=float)
    parser.add_argument('--batch_size', default=12, type=int, help='mini-batch size (default: 12)')
    parser.add_argument('--workers', default=4, help='number of data loading workers (default: 4)', type=int)
    parser.add_argument('--epochs', default=30, help='number of total epochs', type=int)
    parser.add_argument('--test', default=False, help='perform on the test phase', type=bool)
    parser.add_argument('--start_epoch', default=0, help='start epoch if restart', type=int)
    parser.add_argument('--interval', default=10, help='display interval', type=int)
    parser.add_argument('--load_name', default='best', help='loading model number', type=str)
    cfg = parser.parse_args()
    print(cfg)
    if cfg.test:
        print(test_model(cfg, to_file=True))
    else:
        train_model(cfg)
#learning schedule
#test
#logger

if __name__ == '__main__':
    main()