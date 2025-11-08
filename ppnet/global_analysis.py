import os
import re
from argparse import Namespace
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .helpers import makedir
from .find_nearest import find_k_nearest_patches_to_prototypes
from .log import create_logger
from .preprocess import preprocess_input_function


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end,
                                          color=(0, 255, 255), markers=None):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index) + '_prototype-img-original.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    if markers is not None:
        for marker in markers:
            cv2.circle(p_img_bgr, (int(marker[0]), int(marker[1])), 5, (0, 0, 255), -1)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255

    plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def run_analysis(args: Namespace):
    # Choose device automatically
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS backend (Apple Silicon).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA.")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no GPU detected).")

    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    model_path = os.path.abspath(args.model)
    model_base_architecture, experiment_run, _, model_name = re.split(r'\\|/', model_path)[-4:]
    start_epoch_number = int(re.search(r'\d+', model_name).group(0))
    if model_base_architecture == 'pruned_prototypes':
        model_base_architecture, experiment_run = re.split(r'\\|/', model_path)[-6:-4]
        model_name = f'pruned_{model_name}'

    save_analysis_path = os.path.join(args.out, model_base_architecture, experiment_run, model_name, 'global')
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))

    log(f'\nLoad model from: {args.model}')
    log(f'Model epoch: {start_epoch_number}')
    log(f'Model base architecture: {model_base_architecture}')
    log(f'Experiment run: {experiment_run}\n')

    # Load model to correct device
    ppnet = torch.load(args.model, map_location=device)
    ppnet = ppnet.to(device)

    # ❌ No DataParallel on MPS
    if torch.cuda.is_available():
        ppnet_multi = torch.nn.DataParallel(ppnet)
    else:
        ppnet_multi = ppnet  # use single-device mode

    img_size = ppnet.img_size
    batch_size = 100

    # Build dataloaders
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'train'), transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'test'), transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # Output dirs
    root_dir_for_saving_train_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'train')
    root_dir_for_saving_test_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'test')
    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)

    # Prototypes bounding boxes
    load_img_dir = os.path.join(os.path.dirname(args.model), 'img')
    assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}', 'bb.npy'))

    for j in tqdm(range(ppnet.num_prototypes), desc='Saving learned prototypes'):
        for target_dir in [root_dir_for_saving_train_images, root_dir_for_saving_test_images]:
            makedir(os.path.join(target_dir, str(j)))
            save_prototype_original_img_with_bbox(
                load_img_dir=load_img_dir,
                fname=os.path.join(target_dir, str(j), 'prototype_bbox.png'),
                epoch=start_epoch_number,
                index=j,
                bbox_height_start=prototype_info[j][1],
                bbox_height_end=prototype_info[j][2],
                bbox_width_start=prototype_info[j][3],
                bbox_width_end=prototype_info[j][4],
                color=(0, 255, 255)
            )

    # Save nearest prototypes
    for mode, loader, root_dir in [('train', train_loader, root_dir_for_saving_train_images),
                                   ('test', test_loader, root_dir_for_saving_test_images)]:
        log(f'\nSaving nearest prototypes of {mode} set')
        find_k_nearest_patches_to_prototypes(
            dataloader=loader,
            prototype_network_parallel=ppnet_multi,
            k=args.top_imgs + (1 if mode == 'train' else 0),
            preprocess_input_function=preprocess_input_function,
            full_save=True,
            root_dir_for_saving_images=root_dir,
            log=log
        )

    logclose()
