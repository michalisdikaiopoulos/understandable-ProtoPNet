import os
import glob
import shutil
import re
import random
import copy
from collections import defaultdict
from tqdm import tqdm
from argparse import Namespace
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .helpers import makedir, find_high_activation_crop
from .log import create_logger
from .preprocess import mean, std, undo_preprocess_input_function
from .alignment_analysis import alignment_score, save_alignment_matrix


def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(load_img_dir, fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index) + '_prototype-img.png'))
    plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_self_activation(load_img_dir, fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index) + '_prototype-img-original_with_self_act.png'))
    plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index) + '_prototype-img-original.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    # plt.imshow(p_img_rgb)
    plt.imsave(fname, p_img_rgb)


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.axis('off')
    plt.imsave(fname, img_rgb_float)


def run_analysis(args: Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if not os.path.isdir(args.img):
        # Run analysis on a single image
        _run_analysis_on_image(args)
    else:
        # Run analysis on multiple images
        random.seed(1)
        img_filepaths = [f for ext in ['png', 'jpg', 'jpeg'] for f in glob.glob(os.path.join(args.img, f'*/*.{ext}'), recursive=True)]
        # Group by class
        img_filepaths_by_class = defaultdict(list)
        for f in img_filepaths:
            class_name = os.path.basename(os.path.dirname(f))
            img_filepaths_by_class[class_name].append(f)
        # Randomly sample images
        num_classes = len(img_filepaths_by_class)
        num_evaluations = max(num_classes, 100)
        idx = 0
        for class_name, samples in img_filepaths_by_class.items():
            imgs = random.sample(samples, num_evaluations // num_classes)
            for img in imgs:
                print(f'\n\n----------------------------\n[{idx+1}/{num_evaluations}] Evaluating {img}')
                args.img = img
                _run_analysis_on_image(args)
                idx += 1


def _run_analysis_on_image(args: Namespace):
    # Compute params
    img_path = os.path.abspath(args.img)  # ./datasets/celeb_a/gender/test/Male/1.jpg
    img_class, img_id = re.split(r'\\|/', img_path)[-2:]
    img_id, _ = os.path.splitext(img_id)
    img_id = int(img_id)

    dataset_split_path = os.path.dirname(os.path.dirname(img_path))
    dataset_path = os.path.dirname(dataset_split_path)

    model_path = os.path.abspath(args.model)  # ./saved_models/vgg19/003/checkpoints/10_18push0.7822.pth
    model_base_architecture, experiment_run, _, model_name = re.split(r'\\|/', model_path)[-4:]
    if model_base_architecture == 'pruned_prototypes':
        model_base_architecture, experiment_run = re.split(r'\\|/', model_path)[-6:-4]
        model_name = f'pruned_{model_name}'
    start_epoch_number = int(re.search(r'\d+', model_name).group(0))

    save_analysis_path = os.path.join(args.out, model_base_architecture, experiment_run, model_name, 'local', img_class, str(img_id))
    if os.path.exists(save_analysis_path):
        shutil.rmtree(save_analysis_path)
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    log(f'\nLoad model from: {args.model}')
    log(f'Model epoch: {start_epoch_number}')
    log(f'Model base architecture: {model_base_architecture}')
    log(f'Experiment run: {experiment_run}')
    log(f'Output path: {os.path.abspath(save_analysis_path)}\n')

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ppnet = torch.load(args.model, map_location=device)
    ppnet = ppnet.to(device)
    ppnet_multi = ppnet  # no DataParallel on MPS

    img_pil = Image.open(args.img)
    img_size = getattr(ppnet_multi, "img_size", ppnet.img_size)
    prototype_shape = getattr(ppnet_multi, "prototype_shape", ppnet.prototype_shape)
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    normalize = transforms.Normalize(mean=mean, std=std)
    dataset = datasets.ImageFolder(dataset_split_path)

    # Load part annotations
    part_locs = pd.read_csv(os.path.join(dataset_path, 'part_locs.csv'))
    part_locs = part_locs[part_locs.image_id == img_id].drop('image_id', axis=1).set_index('part_name').copy()
    part_locs['x'] = (part_locs['x'] * (img_size / img_pil.width)).astype(int)  # Rescale part locations to match input size
    part_locs['y'] = (part_locs['y'] * (img_size / img_pil.height)).astype(int)
    assert np.all(part_locs[['x', 'y']] <= img_size), 'Part locations are outside of image boundaries'

    # SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(os.path.dirname(args.model), 'img')
    assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}', 'bb.npy'))
    prototype_img_identity = prototype_info[:, -1]
    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' classes')

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect strongly to their respective classes\n')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes\n')

    # load the test image and forward it through the network
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    images_test = img_variable.to(device)
    labels_test = torch.tensor([ dataset.class_to_idx[img_class] ])

    logits, min_distances = ppnet_multi(images_test)
    _, distances = ppnet.push_forward(images_test)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    tables = []
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))

    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    log('Predicted class: ' + str(predicted_cls))
    log('Correct class: ' + str(correct_cls) + '\n')
    original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'), images_test, idx)

    # MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    out_dir = os.path.join(save_analysis_path, 'most_activated_prototypes')
    makedir(out_dir)
    top_prototypes = min(args.top_prototypes, ppnet.num_prototypes)
    alignment_matrix = pd.DataFrame(index=reversed(sorted_indices_act[-top_prototypes:]).cpu().numpy(), columns=part_locs.index)
    for i in tqdm(range(1, top_prototypes + 1), desc='Computing most activated prototypes'):
        save_prototype(load_img_dir, os.path.join(out_dir,  f'top-{i}_prototype_patch.png'), start_epoch_number, sorted_indices_act[-i].item())
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(out_dir, f'top-{i}_prototype_bbox.png'),
            epoch=start_epoch_number,
            index=sorted_indices_act[-i].item(),
            bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
            bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
            bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
            bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
            color=(0, 255, 255)
        )
        save_prototype_self_activation(load_img_dir, os.path.join(out_dir, f'top-{i}_prototype_activation.png'), start_epoch_number, sorted_indices_act[-i].item())
        with open(os.path.join(out_dir, f'top-{i}_info.txt'), 'w') as f:
            f.write('prototype index: {0}\n'.format(sorted_indices_act[-i].item()))
            f.write('prototype class: {0}\n'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
            if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
                f.write('prototype connection: {0}\n'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
            f.write('activation value (similarity score): {0:.4f}\n'.format(array_act[-i]))
            f.write('last layer connection with predicted class: {0:.4f}\n'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
        # Show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
        plt.axis('off')
        plt.imsave(os.path.join(out_dir, f'top-{i}_target_patch.png'), high_act_patch)
        imsave_with_bbox(fname=os.path.join(out_dir, f'top-{i}_target_bbox.png'),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        # Show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        plt.axis('off')
        plt.imsave(os.path.join(out_dir, f'top-{i}_target_activations.png'), overlayed_img)
        # Compute alignment matrix
        alignment_matrix.loc[sorted_indices_act[-i].item(), :] = alignment_score(part_locs, high_act_patch_indices)
    # Plot alignment matrix
    save_alignment_matrix(os.path.join(out_dir, f'top-prototypes_alignment_matrix'), alignment_matrix)
    # PROTOTYPES FROM TOP-k CLASSES
    k = min(args.top_classes, len(dataset.classes))
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i, c in enumerate(topk_classes.detach().cpu().numpy()):
        class_dir = os.path.join(save_analysis_path, 'class_prototypes', f'top-{i+1}_class')
        makedir(class_dir)
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)
        prototype_cnt = 1
        reversed_indices = list(reversed(sorted_indices_cls_act.detach().cpu().numpy()))
        alignment_matrix = pd.DataFrame(index=class_prototype_indices[reversed_indices], columns=part_locs.index)
        for j in tqdm(reversed_indices, desc=f'Computing prototypes of top-{i+1} class'):
            prototype_index = class_prototype_indices[j]
            save_prototype(load_img_dir, os.path.join(class_dir, f'top-{prototype_cnt}_prototype_patch.png'), start_epoch_number, prototype_index)
            save_prototype_original_img_with_bbox(
                load_img_dir=load_img_dir,
                fname=os.path.join(class_dir, f'top-{prototype_cnt}_prototype_bbox.png'),
                epoch=start_epoch_number,
                index=prototype_index,
                bbox_height_start=prototype_info[prototype_index][1],
                bbox_height_end=prototype_info[prototype_index][2],
                bbox_width_start=prototype_info[prototype_index][3],
                bbox_width_end=prototype_info[prototype_index][4],
                color=(0, 255, 255)
            )
            save_prototype_self_activation(load_img_dir, os.path.join(class_dir, f'top-{prototype_cnt}_prototype_activation.png'), start_epoch_number, prototype_index)
            with open(os.path.join(class_dir, f'top-{prototype_cnt}_info.txt'), 'w') as f:
                f.write('prototype index: {0}\n'.format(prototype_index))
                f.write('prototype class: {0}\n'.format(prototype_img_identity[prototype_index]))
                f.write('prototype class logits: {0:.4f}\n'.format(topk_logits[i]))
                if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                    f.write('prototype connection: {0}\n'.format(prototype_max_connection[prototype_index]))
                f.write('activation value (similarity score): {0:.4f}\n'.format(prototype_activations[idx][prototype_index]))
                f.write('last layer connection: {0:.4f}\n'.format(ppnet.last_layer.weight[c][prototype_index]))
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
            # Show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
            plt.axis('off')
            plt.imsave(os.path.join(class_dir, f'top-{prototype_cnt}_target_patch.png'), high_act_patch)
            imsave_with_bbox(fname=os.path.join(class_dir, f'top-{prototype_cnt}_target_bbox.png'),
                             img_rgb=original_img,
                             bbox_height_start=high_act_patch_indices[0],
                             bbox_height_end=high_act_patch_indices[1],
                             bbox_width_start=high_act_patch_indices[2],
                             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            # Show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            plt.axis('off')
            plt.imsave(os.path.join(class_dir, f'top-{prototype_cnt}_target_activation.png'), overlayed_img)
            prototype_cnt += 1
            # Compute alignment matrix
            alignment_matrix.loc[prototype_index, :] = alignment_score(part_locs, high_act_patch_indices)
        # Plot alignment matrix
        save_alignment_matrix(os.path.join(class_dir, f'top-prototypes_alignment_matrix'), alignment_matrix)

    if predicted_cls == correct_cls:
        log('Prediction is correct.')
    else:
        log('Prediction is wrong.')

    logclose()
