# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for RelViT. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import json
import re
import glob
import os.path as osp
import collections

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate

import utils
from .datasets import register


def label_to_onehot(label, num_class):
    # label: [N]
    onehot = torch.zeros(num_class).to(label)
    for i in label:
        onehot[i] = 1
    return onehot


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def prep_ans(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


@register('gqa')
class GQA(Dataset):
    def __init__(self, root_dir=None, split='train', eval_mode=0, **kwargs) -> None:
        self.eval_mode = eval_mode
        self.FRCN_FEAT_SIZE = utils.MCAN_GQA_PARAMS['FRCN_FEAT_SIZE']
        self.GRID_FEAT_SIZE = utils.MCAN_GQA_PARAMS['GRID_FEAT_SIZE']
        self.BBOX_FEAT_SIZE = utils.MCAN_GQA_PARAMS['BBOX_FEAT_SIZE']
        self.root_dir = root_dir
        self.split = split
        self.dict_json = osp.join(self.root_dir, 'dicts.json')

        if self.eval_mode:
            # object/edge/degree
            fn = osp.join(self.root_dir, 'raw', 'questions1.2', '{}_sys_reduced_questions.json'.format(self.split))
            fn_concept = osp.join(self.root_dir, 'raw', 'questions1.2', 'train_sys_reduced_concepts.json'.format(self.split))
        else:
            if self.split is 'val':
                split = 'val'
            else:
                split = self.split
            fn = osp.join(self.root_dir, 'raw', 'questions1.2', '{}_balanced_questions.json'.format(split))
            fn_concept = osp.join(self.root_dir, 'raw', 'questions1.2', 'train_balanced_concepts.json'.format(self.split))
        with open(fn, 'r') as f:
            self.ques_dict = json.load(f)
        self.qid_list = list(self.ques_dict.keys())
        with open(fn_concept, 'r') as f:
            self.concept_dict = json.load(f)

        def img_feat_path_load(path_list):
            iid_to_path = {}
            for ix, path in enumerate(path_list):
                iid = path.split('/')[-1].split('.')[0]
                iid_to_path[iid] = path

            return iid_to_path
        self.iid_to_img_path = img_feat_path_load(glob.glob(osp.join(
            self.root_dir,
            'images',
            '*.jpg'
        )))
        self.data_size = self.ques_dict.__len__()
        self.num_concept = 1615

        # Tokenize
        # self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(self.dict_json, True)
        self.token_to_ix, _, max_token = self.tokenize(self.dict_json, True)
        self.token_size = self.token_to_ix.__len__()

        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat(self.dict_json)
        self.ans_size = self.ans_to_ix.__len__()

        # For RGB
        self.pix_mean = (0.485, 0.456, 0.406)
        self.pix_std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.pix_mean, std=self.pix_std)
        ])
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


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx, raw_q=False)
        img = Image.open(self.iid_to_img_path[iid]).convert('RGB')
        img = self.transform(img)
        second_im = self.densecl_aug(img)
        third_im = self.densecl_aug(img)

        concept = np.array(self.load_concept(idx), dtype=np.int64)
        concept = label_to_onehot(torch.from_numpy(concept).long(), self.num_concept)

        # len(ret), ret[-3]
        # 7, True
        ret = []
        ret.append(img)
        ret.append(second_im)
        ret.append(third_im)
        ret.append(torch.from_numpy(ques_ix_iter))
        ret.append(True)
        ret.append(torch.from_numpy(ans_iter))
        ret.append(concept)

        return tuple(ret)

    def tokenize(self, json_file, use_glove):
        token_to_ix, max_token = json.load(open(json_file, 'r'))[2:]
        # spacy_tool = None
        # if use_glove:
        #     spacy_tool = en_vectors_web_lg.load()

        # pretrained_emb = []
        # for word in token_to_ix:
        #     if use_glove:
        #         pretrained_emb.append(spacy_tool(word).vector)
        # pretrained_emb = np.array(pretrained_emb)

        pretrained_emb = None
        return token_to_ix, pretrained_emb, max_token

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))[:2]

        return ans_to_ix, ix_to_ans

    def load_concept(self, idx):
        qid = self.qid_list[idx]
        if self.split == 'train':
            ret = self.concept_dict[qid]
        else:
            ret = []
        return ret

    def load_ques_ans(self, idx, raw_q=False):

        qid = self.qid_list[idx]
        iid = self.ques_dict[qid]['imageId']

        ques = self.ques_dict[qid]['question']
        if raw_q:
            ques_ix_iter = ques
        else:
            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=self.max_token)

        # process answers
        ans = self.ques_dict[qid]['answer']
        ans_iter = self.proc_ans(ans, self.ans_to_ix)

        return ques_ix_iter, ans_iter, iid

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat

    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans = prep_ans(ans)
        ans_ix[0] = ans_to_ix[ans]

        return ans_ix


def collate_gqa(batch):
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

    # len(ret), ret[-3]
    # 7, True
    if len(batch[0][0].shape) == 2:
        return default_collate(batch)
    else:
        list_ims = []
        list_second_ims = []
        list_third_ims = []
        list_qs = []
        list_xs = []
        list_as = []
        list_cs = []
        for b in batch:
            list_ims.append(b[0])
            list_second_ims.append(b[1])
            list_third_ims.append(b[2])
            list_qs.append(b[3])
            list_xs.append(b[4])
            list_as.append(b[5])
            list_cs.append(b[6])
        list_ims = torch.stack(_pad_tensor(list_ims))
        list_second_ims = torch.stack(_pad_tensor(list_second_ims))
        list_third_ims = torch.stack(_pad_tensor(list_third_ims))
        if list_xs[0]:
            list_qs = torch.stack(list_qs)
        list_as = torch.stack(list_as)
        list_cs = torch.stack(list_cs)
        return list_ims, list_second_ims, list_third_ims, list_qs, list_xs, list_as, list_cs
