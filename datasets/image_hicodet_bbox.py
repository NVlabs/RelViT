# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for RelViT. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import os
import os.path as osp
import json
import pickle
import numpy as np
from PIL import Image, ImageFilter
import sklearn.metrics

import cv2
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from detectron2
from detectron2.structures import Boxes
from detectron2.data import transforms as T

import utils
from .datasets import register


def label_to_onehot(label, num_class):
    # label: [N]
    onehot = torch.zeros(num_class).to(label)
    for i in label:
        onehot[i] = 1
    return onehot


@register('hicodet')
class HICODet(Dataset):
    def __init__(self, image_size=256, box_size=256, eval_mode=0, **kwargs) -> None:
        self.eval_mode = eval_mode
        im_dir = kwargs.get('im_dir')
        self.split = kwargs.get('split') # train or test
        self._root = osp.join(im_dir, 'hico_20160224_det', 'images', '{}2015'.format(self.split))
        # rare
        if self.eval_mode == 1:
            anno_file = osp.join(im_dir, 'hico_20160224_det', 'sys_vcl_rare_instances_{}2015.json'.format(self.split))
        # non-rare
        elif self.eval_mode == 2:
            anno_file = osp.join(im_dir, 'hico_20160224_det', 'sys_vcl_nonrare_instances_{}2015.json'.format(self.split))
        else:
            anno_file = osp.join(im_dir, 'hico_20160224_det', 'instances_{}2015.json'.format(self.split))

        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80
        self.num_interaction_cls = 600
        self.num_action_cls = 117
        self._anno_file = anno_file

        # Load annotations
        self._load_annotation_and_metadata(anno)

        self.pix_mean = (0.485, 0.456, 0.406)
        self.pix_std = (0.229, 0.224, 0.225)
        # detectron2-style data augmentation
        sample_style = 'range'
        augmentations = [T.ResizeShortestEdge(image_size, int(image_size * 2), sample_style)]
        if kwargs.get('augment') or kwargs.get('augment_plus'):
            augmentations.append(
                T.RandomFlip(
                    horizontal=True,
                    vertical=False,
                )
            )
        if kwargs.get('augment_plus'):
            self.photo_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        else:
            self.photo_aug = None
        self.augmentations = T.AugmentationList(augmentations)

        # For two-view denseCL
        # https://github.com/WXinlong/DenseCL/blob/main/configs/selfsup/densecl/densecl_coco_800ep.py
        self.densecl_aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )]),
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.GaussianBlur(
                        # https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py#L20
                        kernel_size=(int(0.1*224)//2)*2+1,
                        sigma=(0.1, 2.0)
                    )]),
                p=0.5
            ),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        """
        ind = self._idx[i]
        im = cv2.imread(osp.join(self._root, self._filenames[ind]))
        assert im is not None
        # BGR to RGB
        im = im[:, :, ::-1]
        # TODO: only keeps the first bbox pair
        sub_bbox = torch.Tensor([self._anno[ind]['boxes_h'][0]])
        obj_bbox = torch.Tensor([self._anno[ind]['boxes_o'][0]])
        union_bbox = torch.cat((torch.min(sub_bbox[0, :2], obj_bbox[0, :2]), torch.max(sub_bbox[0, 2:], obj_bbox[0, 2:]))).unsqueeze(0)
        boxes = torch.stack((sub_bbox, obj_bbox, union_bbox), dim=0)
        aug_input = T.AugInput(im, boxes=boxes)
        transforms = self.augmentations(aug_input)
        im, boxes = aug_input.image, torch.as_tensor(aug_input.boxes)
        im = im.astype(np.float32)

        def to_tensor(im, pix_mean, pix_std, normalize=True):
            if normalize:
                for i in range(3):
                    im[:, :, i] = (im[:, :, i] / 255. - pix_mean[i]) / pix_std[i]
            im = torch.as_tensor(np.ascontiguousarray(im.transpose(2, 0, 1))).float()
            return im

        second_im = im.copy()
        third_im = im.copy()
        # Augment&tensorize the main view
        if self.photo_aug is not None:
            # color jittering of the input image
            im = np.array(self.photo_aug(Image.fromarray(im.astype(np.uint8))), dtype=np.float32)
        im = to_tensor(im, self.pix_mean, self.pix_std)

        # Augment&tensorize the second view
        second_im = np.array(self.densecl_aug(Image.fromarray(second_im.astype(np.uint8))), dtype=np.float32)
        second_im = to_tensor(second_im, self.pix_mean, self.pix_std)

        # Augment&tensorize the third view
        third_im = np.array(self.densecl_aug(Image.fromarray(third_im.astype(np.uint8))), dtype=np.float32)
        third_im = to_tensor(third_im, self.pix_mean, self.pix_std)

        hoi = label_to_onehot(torch.Tensor(self._anno[ind]['hoi']).long(), self.num_interaction_cls)
        verb = label_to_onehot(torch.Tensor(self._anno[ind]['verb']).long(), self.num_action_cls)
        object = label_to_onehot(torch.Tensor(self._anno[ind]['object']).long(), self.num_object_cls)
        return im, second_im, third_im, boxes, hoi, verb, object

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idx = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idx.remove(empty_idx)

        num_anno = [0 for _ in range(self.num_interaction_cls)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idx
        self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._image_sizes = f['size']
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']

    def _ood_split(self):
        pass

def compute_map_hico(y_true, y_score, easy=False, hard=False, rare_only=False):
    unseen_hoi_nonrare = np.array([38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61,
                    457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73,
                            159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346,
                                    456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572,
                                                529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329, 246, 173, 506,
                                                        383, 93, 516, 64])

    unseen_hoi_rare = np.array([509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416,
                    389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596, 345, 189,
                            205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229, 158, 195,
                                    238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188, 216, 597,
                                                77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104, 55, 50,
                                                        198, 168, 391, 192, 595, 136, 581])
    rare_hoi = np.load('./cache/rare.npy')
    if easy:
        index = unseen_hoi_rare
    if hard:
        index = unseen_hoi_nonrare
    if rare_only:
        index = rare_hoi
    if easy or hard or rare_only:
        y_true = y_true[:, index]
        y_score = y_score[:, index]
    meter = utils.AveragePrecisionMeter(algorithm='AUC', output=torch.Tensor(y_score), labels=torch.Tensor(y_true))
    return meter.eval().mean()


def collate_images_boxes_dict(batch):
    def _pad_tensor(tensor_list):
        max_imh, max_imw = -1, -1
        for tensor_i in tensor_list:
            # import pdb; pdb.set_trace()
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for idx, tensor_i in enumerate(tensor_list):
            pad_tensor_i = tensor_i.new_full(list(tensor_i.shape[:-2]) + [max_imh, max_imw], 0)
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            pad_tensor_i[..., :imh, :imw].copy_(tensor_i)
            tensor_list[idx] = pad_tensor_i
        return tensor_list
    list_ims = []
    list_second_ims = []
    list_third_ims = []
    list_boxes = []
    list_hois = []
    list_verbs = []
    list_objects = []
    for b in batch:
        list_ims.append(b[0])
        list_second_ims.append(b[1])
        list_third_ims.append(b[2])
        list_boxes.append(b[3])
        list_hois.append(b[4])
        list_verbs.append(b[5])
        list_objects.append(b[6])

    list_ims = _pad_tensor(list_ims)
    list_second_ims = _pad_tensor(list_second_ims)
    list_third_ims = _pad_tensor(list_third_ims)
    return torch.stack(list_ims), torch.stack(list_second_ims), torch.stack(list_third_ims), torch.stack(list_boxes), torch.stack(list_hois), torch.stack(list_verbs), torch.stack(list_objects)
