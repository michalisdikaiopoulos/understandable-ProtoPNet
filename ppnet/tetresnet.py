import os
import re
import time
from tqdm import tqdm
from argparse import Namespace
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models

from .helpers import set_seed
from .log import create_logger
from .preprocess import mean, std


def _train_or_test_resnet(model, dataloader, optimizer=None, log=print):
    '''
    Train or test a ResNet model
    model: the multi-gpu model
    dataloader: data loader
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0

    for i, (image, label) in enumerate(tqdm(dataloader)):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output = model(input)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        # compute gradient and do SGD step
        if is_train:
            optimizer.zero_grad()
            cross_entropy.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted

    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end - start))
    log('\tcross ent: \t{0:.5f}'.format(total_cross_entropy / n_batches))
    log('\taccu: \t\t{0:.5f}%'.format(n_correct / n_examples * 100))

    return n_correct / n_examples


def train_resnet(model, dataloader, optimizer, log=print):
    assert(optimizer is not None)
    log('train')
    model.train()
    return _train_or_test_resnet(model=model, dataloader=dataloader, optimizer=optimizer, log=log)


def test_resnet(model, dataloader, log=print):
    log('test')
    model.eval()
    return _train_or_test_resnet(model=model, dataloader=dataloader, optimizer=None, log=log)


def construct_resnet(base_architecture='resnet34', pretrained=True, num_classes=200):
    '''
    Construct a ResNet model
    '''
    if base_architecture == 'resnet18':
        resnet = models.resnet18(pretrained=pretrained)
    elif base_architecture == 'resnet34':
        resnet = models.resnet34(pretrained=pretrained)
    elif base_architecture == 'resnet50':
        resnet = models.resnet50(pretrained=pretrained)
    elif base_architecture == 'resnet101':
        resnet = models.resnet101(pretrained=pretrained)
    elif base_architecture == 'resnet152':
        resnet = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported architecture: {base_architecture}')
    
    # Replace the final fully connected layer
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    return resnet


def save_resnet_model(model, model_dir, model_name, accu, log=print):
    '''
    Save ResNet model
    '''
    torch.save(obj=model.state_dict(), f=os.path.join(model_dir, f'{model_name}.pth'))
    log(f'\tSaved model: {model_name}.pth (accuracy: {accu:.5f})')


def run_resnet_training(args: Namespace):
    # Init environment
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('GPUs:', os.environ['CUDA_VISIBLE_DEVICES'])

    base_architecture = re.match('^[a-z0-9]*', args.architecture).group(0)
    model_dir = os.path.join('./saved_models', args.architecture, args.exp_name + '_resnet')
    if os.path.exists(model_dir):
        print(f'Warning: model directory "{args.exp_name}_resnet" already exists, overwriting...')
    else:
        os.makedirs(model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    with open(os.path.join(model_dir, 'args.yaml'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    # load the data
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    normalize = T.Normalize(mean=mean, std=std)

    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        T.Compose([
            T.RandomAffine(degrees=20, shear=15),
            T.RandomPerspective(distortion_scale=0.25, p=0.25),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)
    
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        T.Compose([
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(args.batch_size))

    # construct the model
    resnet = construct_resnet(base_architecture=base_architecture,
                              pretrained=True,
                              num_classes=len(train_dataset.classes))
    resnet = resnet.cuda()
    resnet_multi = torch.nn.DataParallel(resnet)

    # define optimizer - using same learning rates as joint training in original code
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # train the model
    log('\nStart training ResNet')
    best_accu = 0.0
    for epoch in range(args.epochs + 1):
        log('epoch: \t{0}'.format(epoch))

        if epoch > 0:  # Skip training on epoch 0, just evaluate
            _ = train_resnet(model=resnet_multi, dataloader=train_loader, optimizer=optimizer, log=log)
            lr_scheduler.step()

        if epoch % args.test_interval == 0:
            accu = test_resnet(model=resnet_multi, dataloader=test_loader, log=log)
            
            # Save model
            save_resnet_model(model=resnet, model_dir=model_dir, 
                            model_name=f'resnet_{epoch:03d}', accu=accu, log=log)
            
            if accu > best_accu:
                best_accu = accu
                save_resnet_model(model=resnet, model_dir=model_dir, 
                                model_name='resnet_best', accu=accu, log=log)

        log('------------------------------------------\n')
    
    log(f'\nBest accuracy: {best_accu:.5f}')
    logclose()
