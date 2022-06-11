# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for RelViT. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import argparse
import os
from re import L
import yaml
import numpy as np
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from tensorboardX import SummaryWriter

from tqdm import tqdm

import datasets
import models
import utils
from datasets.image_hicodet_bbox import collate_images_boxes_dict, compute_map_hico

from detectron2.structures import Boxes


def gather_score_label(dist, loader, current_idx, y_true, y_score, ddp=False):
    if ddp:
        # all_gather and compute mAP
        size = y_true.size(-1)
        current_idx_list = [torch.zeros(1).long().cuda(non_blocking=True) for _ in range(args.world_size)]
        y_true_list = [torch.zeros((len(loader.dataset), size)).long().cuda(non_blocking=True) for _ in range(args.world_size)]
        y_score_list = [torch.zeros((len(loader.dataset), size)).float().cuda(non_blocking=True) for _ in range(args.world_size)]
        dist.all_gather(current_idx_list, current_idx)
        dist.all_gather(y_true_list, y_true)
        dist.all_gather(y_score_list, y_score)

        final_y_true = np.empty((0, size))
        final_y_score = np.empty((0, size))
        for idx, yt, ys in zip(current_idx_list, y_true_list, y_score_list):
            idx = idx.item()
            yt = yt.detach().cpu().numpy()
            ys = ys.detach().cpu().numpy()
            final_y_true = np.vstack((final_y_true, yt[:idx]))
            final_y_score = np.vstack((final_y_score, ys[:idx]))
    else:
        final_y_true = y_true.detach().cpu().numpy()[:current_idx.item()]
        final_y_score = y_score.detach().cpu().numpy()[:current_idx.item()]
    return final_y_true, final_y_score


def main(config):
    args.gpu = ''#[i for i in range(torch.cuda.device_count())]
    args.train_gpu = [i for i in range(torch.cuda.device_count())]
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus - 1):
        args.gpu += '{},'.format(i)
    args.gpu += '{}'.format(num_gpus - 1)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    utils.set_gpu(args.gpu)
    args.config = config

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.sync_bn = True
        if args.dist_url[-2:] == '{}':
            port = utils.find_free_port()
            args.dist_url = args.dist_url.format(port)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    config = args.config
    svname = args.svname
    if svname is None:
        config_name, _ = os.path.splitext(os.path.basename(args.config_file))
        svname = 'hico'
        svname += '_' + config['model']
        if config['model_args'].get('encoder'):
            svname += '-' + config['model_args']['encoder']
        svname = os.path.join(config_name, config['train_dataset'], svname)
    if not args.test_only:
        svname += '-seed' + str(args.seed)
    if args.tag is not None:
        svname += '_' + args.tag

    sub_dir_name = 'default'
    if args.opts:
        sub_dir_name = args.opts[0]
        split = '#'
        for opt in args.opts[1:]:
            sub_dir_name += split + opt
            split = '#' if split == '_' else '_'
    svname = os.path.join(svname, sub_dir_name)

    if utils.is_main_process() and not args.test_only:
        save_path = os.path.join(args.save_dir, svname)
        utils.ensure_path(save_path, remove=False)
        utils.set_log_path(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        args.writer = writer

        yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

        logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)
    else:
        save_path = None
        writer = None
        args.writer = writer
        logger = None

    #### Dataset ####

    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    import json
    with open(os.path.join(
            config['train_dataset_args']['im_dir'],
            'hico_20160224_det',
            'instances_train2015.json'
        ), 'r') as f:
        correspondance = json.load(f)['correspondence']

    # train
    dataset_configs = config['train_dataset_args']
    train_dataset = datasets.make(config['train_dataset'], eval_mode=config['eval_mode'], **dataset_configs)
    if utils.is_main_process():
        utils.log('train dataset: {} samples'.format(len(train_dataset)))
    if args.distributed:
        args.batch_size = int(ep_per_batch / args.world_size)
        args.batch_size_test = int(ep_per_batch / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    else:
        args.batch_size = ep_per_batch
        args.batch_size_test = ep_per_batch
        args.workers = args.workers

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_images_boxes_dict
    )

    # testing
    dataset_configs = config['test_dataset_args']
    test_dataset = datasets.make(config['test_dataset'], eval_mode=config['eval_mode'], **dataset_configs)
    if utils.is_main_process():
        utils.log('test dataset: {} samples'.format(len(test_dataset)))
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
        collate_fn=collate_images_boxes_dict
    )

    ########

    #### Model and optimizer ####

    if config.get('load'):
        print('loading pretrained model: ', config['load'])
        model = models.load(torch.load(config['load']))
        if config['relvit']:
            model_tea_k = models.load(torch.load(config['load']))
    else:
        model = models.make(config['model'], **config['model_args'])
        if config['relvit']:
            model_tea_k = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            pretrain = config.get('encoder_pretrain').lower()
            if pretrain != 'scratch':
                pretrain_model_path = config['load_encoder'].format(pretrain)
                state_dict = torch.load(pretrain_model_path, map_location='cpu')
                missing_keys, unexpected_keys = model.encoder.encoder.load_state_dict(state_dict, strict=False)
                # from IPython import embed; embed()
                for key in missing_keys:
                    assert key.startswith('g_mlp.') \
                        or key.startswith('proj') \
                        or key.startswith('trans') \
                        or key.startswith('roi_processor') \
                        or key.startswith('roi_dim_processor') \
                        or key.startswith('classifier'), key
                for key in unexpected_keys:
                    assert key.startswith('fc.')
                if utils.is_main_process():
                    utils.log('==> Successfully loaded {} for the enocder.'.format(pretrain_model_path))
        if config['relvit']:
            model_tea_k.load_state_dict(model.state_dict())

        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if config['relvit']:
                model_tea_k = nn.SyncBatchNorm.convert_sync_batchnorm(model_tea_k)

        if utils.is_main_process():
            utils.log(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
        if config['relvit']:
            model_tea_k = torch.nn.parallel.DistributedDataParallel(model_tea_k.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())
        if config['relvit']:
            model_tea_k = torch.nn.DataParallel(model_tea_k.cuda())

    if utils.is_main_process() and not args.test_only:
        utils.log('num params: {}'.format(utils.compute_n_params(model)))
        utils.log('Results will be saved to {}'.format(save_path))

    max_steps = min(len(train_loader), config['train_batches']) * config['max_epoch']
    optimizer, lr_scheduler, update_lr_every_epoch = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], max_steps, **config['optimizer_args']
    )
    assert lr_scheduler is not None
    args.update_lr_every_epoch = update_lr_every_epoch

    if args.test_only:
        filename = args.test_model
        assert os.path.exists(filename)
        ckpt = torch.load(filename, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        if config['relvit']:
            model_tea_k.load_state_dict(ckpt['tea_k_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        best_test_result = ckpt['best_test_result']
        if utils.is_main_process():
            utils.log('==> Sucessfully resumed from a checkpoint {}'.format(filename))
    else:
        ckpt = None
        start_epoch = 0
        best_test_result = 0.0

    ######## MoCo
    if config['relvit']:
        feat_dim = model.module.encoder.encoder.out_dim
        moco = utils.relvit.MoCo(
            model_tea_k.module,
            model.module,
            config['relvit_moco_K'],
            config['relvit_moco_m'],
            feat_dim,
            num_concepts=config['relvit_num_concepts'],
            relvit_mode=config['relvit_mode'],
            num_tokens=config['relvit_num_tokens'])
        moco = moco.cuda()
        if ckpt is not None:
            try:
                moco.load_state_dict(ckpt['moco'])
                utils.log('==> MoCo sucessfully resumed from a checkpoint.')
            except:
                utils.log('==> MoCo is not resumed.')
    else:
        moco = None
        model_tea_k = None
    ########

    ######## Training & Validation

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = best_test_result
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    if args.test_only:
        ret = test(test_loader, model_tea_k if model_tea_k is not None else model, correspondance, 0)
        if ret is None:
            return 0
        else:
            loss_test, map_test = ret
            if utils.is_main_process():
                print('testing result: ', map_test)
            return 0

    for epoch in range(start_epoch, max_epoch):
        # timer_epoch.s()
        # aves = {k: utils.Averager() for k in aves_keys}

        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        ret = train(train_loader, model, optimizer, lr_scheduler, epoch_log, (moco, model_tea_k), correspondance, writer, args)
        if ret is None:
            return 0
        else:
            (loss_train, aux_loss_train), map_train = ret
        import gc; gc.collect()
        torch.cuda.empty_cache()
        if args.update_lr_every_epoch:
            lr_scheduler.step()
        if utils.is_main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('aux_loss_train', aux_loss_train, epoch_log)
            writer.add_scalar('mAP_train', map_train['mAP'], epoch_log)
            if config['eval_mode']:
                writer.add_scalar('mAP_train_verb', map_train['mAP_verb'], epoch_log)
                writer.add_scalar('mAP_train_object', map_train['mAP_obj'], epoch_log)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch_log)

        if (epoch_log % config['save_epoch'] == 0 or epoch_log == config['max_epoch']) and utils.is_main_process():
            filename = os.path.join(save_path, 'train.pth')
            utils.log('==> Saving checkpoint to: ' + filename)
            ckpt = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'tea_k_state_dict': model_tea_k.state_dict() if config['relvit'] else model.state_dict(),
                'moco': moco.state_dict() if moco else None,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_test_result': best_test_result,
            }
            torch.save(ckpt, filename)
        test_res = 0
        if epoch_log % config['eval_epoch'] == 0:
            avg_map_test = 0
            ret = test(test_loader, model_tea_k if model_tea_k is not None else model, correspondance, epoch_log)
            if ret is None:
                return 0
            else:
                loss_test, map_test = ret
            import gc; gc.collect()
            torch.cuda.empty_cache()
            if config['eval_mode']:
                test_res = map_test['mAP']
            else:
                test_res = map_test
            if test_res > best_test_result:
                best_test_result = test_res
            if utils.is_main_process():
                utils.log('test result: loss {:.4f}, mAP: {:.4f}.'.format(loss_test, test_res))
                writer.add_scalar('loss_test', loss_test, epoch_log)
                writer.add_scalar('mAP_test', map_test['mAP'], epoch_log)
                writer.add_scalar('mAP_test_unseen', map_test['mAP_unseen'], epoch_log)
                writer.add_scalar('mAP_test_rare', map_test['mAP_rare'], epoch_log)
                if config['eval_mode']:
                    writer.add_scalar('mAP_test_verb', map_test['mAP_verb'], epoch_log)
                    writer.add_scalar('mAP_test_object', map_test['mAP_obj'], epoch_log)

        if utils.is_main_process():
            utils.log('Best test results so far:')
            utils.log(best_test_result)

        if test_res > max_va and utils.is_main_process():
            max_va = test_res
            filename = os.path.join(save_path, 'best_model.pth')
            ckpt = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'tea_k_state_dict': model_tea_k.state_dict() if config['relvit'] else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_test_result': best_test_result,
            }
            torch.save(ckpt, filename)
        if utils.is_main_process():
            writer.flush()

    if utils.is_main_process():
        logger.close()

def train(train_loader, model, optimizer, lr_scheduler, epoch, moco_tuple, correspondance, writer, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    main_loss_meter = utils.AverageMeter()
    aux_loss_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    intersection_meter = utils.AverageMeter()
    union_meter = utils.AverageMeter()
    target_meter = utils.AverageMeter()
    # mAP
    current_idx = torch.zeros(1).long()
    y_true = torch.zeros((len(train_loader.dataset), train_loader.dataset.num_interaction_cls)).long()
    verb_true = torch.zeros((len(train_loader.dataset), train_loader.dataset.num_action_cls)).long()
    obj_true = torch.zeros((len(train_loader.dataset), train_loader.dataset.num_object_cls)).long()
    y_score = torch.zeros((len(train_loader.dataset), train_loader.dataset.num_interaction_cls)).float()
    verb_score = torch.zeros((len(train_loader.dataset), train_loader.dataset.num_action_cls)).float()
    obj_score = torch.zeros((len(train_loader.dataset), train_loader.dataset.num_object_cls)).float()

    config = args.config

    # train
    model.train()

    if utils.is_main_process():
        args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    end = time.time()
    max_iter = config['max_epoch'] * len(train_loader)
    for batch_idx, (ims, second_ims, third_ims, boxes, hois, verbs, objects) in enumerate(train_loader):
        if batch_idx >= config['train_batches']:
            break
        ims = ims.cuda(non_blocking=True)
        second_ims = second_ims.cuda(non_blocking=True)
        third_ims = third_ims.cuda(non_blocking=True)
        boxes = [[Boxes(b.cuda(non_blocking=True))] for b in boxes]
        data_time.update(time.time() - end)
        hois = hois.cuda(non_blocking=True)
        verbs = verbs.cuda(non_blocking=True)
        objects = objects.cuda(non_blocking=True)
        if config['relvit_concept_use'] == 0:
            concepts = hois
        elif config['relvit_concept_use'] == 1:
            concepts = verbs
        else:
            concepts = objects

        # For Two-view denseCL
        cl_ims = torch.cat([second_ims, third_ims], dim=0)
        cl_boxes = []
        cl_boxes.extend(boxes)
        cl_boxes.extend(boxes)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if config['relvit']:
                if config['eval_mode']:
                    logits_verb, logits_object = model(ims, boxes)
                    logits = utils.hico_logit_conversion_to_hoi(logits_verb, logits_object, correspondance).detach()
                else:
                    logits = model(ims, boxes)
                B = cl_ims.size(0) // 2
                with torch.no_grad():
                    d1, attn_v_k = moco_tuple[1](cl_ims, cl_boxes, True)
                d2, attn_v_q = model(cl_ims, cl_boxes, True)
                aux_loss = moco_tuple[0](attn_v_k, attn_v_q, B, config, args, concepts) + d1[0].mean()*0 + d1[1].mean()*0 + d2[0].mean()*0 + d2[1].mean()*0
            else:
                if config['eval_mode']:
                    (logits_verb, logits_object) = model(ims, boxes)
                    logits = utils.hico_logit_conversion_to_hoi(logits_verb, logits_object, correspondance).detach()
                else:
                    logits = model(ims, boxes)
                aux_loss = torch.zeros(1).to(logits)

            if config['eval_mode']:
                loss = F.binary_cross_entropy_with_logits(logits_verb, verbs.float(), pos_weight=torch.ones(logits_verb.size(1)).to(logits_verb)*10) + \
                        F.binary_cross_entropy_with_logits(logits_object, objects.float(), pos_weight=torch.ones(logits_object.size(1)).to(logits_object)*10) + \
                        config['relvit_weight'] * aux_loss
            else:
                loss = F.binary_cross_entropy_with_logits(logits, hois.float(), pos_weight=torch.ones(logits.size(1)).to(logits)*10) + config['relvit_weight'] * aux_loss
            y_score[current_idx.item():current_idx.item()+logits.size(0)] = logits.detach().cpu()
            y_true[current_idx.item():current_idx.item()+logits.size(0)] = hois.detach().cpu()
            if config['eval_mode']:
                verb_score[current_idx.item():current_idx.item()+logits.size(0)] = logits_verb.detach().cpu()
                obj_score[current_idx.item():current_idx.item()+logits.size(0)] = logits_object.detach().cpu()
                verb_true[current_idx.item():current_idx.item()+logits.size(0)] = verbs.detach().cpu()
                obj_true[current_idx.item():current_idx.item()+logits.size(0)] = objects.detach().cpu()
            current_idx += logits.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lrs = lr_scheduler.get_last_lr()
        if not args.update_lr_every_epoch:
            lr_scheduler.step()

        n = ims.size(0)
        if args.multiprocessing_distributed:
            # TODO: all gather y_score/true
            loss = loss * n  # not considering ignore pixels
            count = hois.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            n = count.item()
            loss = loss / n

        loss_meter.update(loss.item(), ims.size(0))
        aux_loss_meter.update(aux_loss.item(), ims.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + batch_idx + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (batch_idx + 1) % config['print_freq'] == 0 and utils.is_main_process():
            utils.log(
                '{} Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'Loss {loss_meter.val:.4f} '
                'Aux Loss {aux_loss_meter.val:.4f} '
                'lr {lr:.6f}'.format(
                    'Train',
                    epoch, config['max_epoch'], batch_idx + 1, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    aux_loss_meter=aux_loss_meter,
                    lr=lrs[0]
                )
            )
    final_y_true, final_y_score = gather_score_label(dist, train_loader, current_idx.cuda(), y_true.cuda(), y_score.cuda(), args.multiprocessing_distributed)
    map = compute_map_hico(final_y_true, final_y_score)
    if config['eval_mode']:
        final_verb_true, final_verb_score = gather_score_label(dist, train_loader, current_idx.cuda(), verb_true.cuda(), verb_score.cuda(), args.multiprocessing_distributed)
        final_obj_true, final_obj_score = gather_score_label(dist, train_loader, current_idx.cuda(), obj_true.cuda(), obj_score.cuda(), args.multiprocessing_distributed)
        map_verb = compute_map_hico(final_verb_true, final_verb_score)
        map_obj = compute_map_hico(final_obj_true, final_obj_score)
    if utils.is_main_process():
        utils.log('{} result at epoch [{}/{}]: loss {:.4f}, mAP {:.4f}.'.format('Train', epoch, config['max_epoch'], loss_meter.avg, map))
    if config['eval_mode']:
        return (loss_meter.avg, aux_loss_meter.avg), {'mAP': map, 'mAP_verb': map_verb, 'mAP_obj': map_obj}
    else:
        return (loss_meter.avg, aux_loss_meter.avg), {'mAP': map}

def test(test_loader, model, correspondance, epoch_log):
    # eval
    model.eval()

    config = args.config
    loss_meter = utils.AverageMeter()
    # mAP
    current_idx = torch.zeros(1).long()
    y_true = torch.zeros((len(test_loader.dataset), test_loader.dataset.num_interaction_cls)).long()
    verb_true = torch.zeros((len(test_loader.dataset), test_loader.dataset.num_action_cls)).long()
    obj_true = torch.zeros((len(test_loader.dataset), test_loader.dataset.num_object_cls)).long()
    y_score = torch.zeros((len(test_loader.dataset), test_loader.dataset.num_interaction_cls)).float()
    verb_score = torch.zeros((len(test_loader.dataset), test_loader.dataset.num_action_cls)).float()
    obj_score = torch.zeros((len(test_loader.dataset), test_loader.dataset.num_object_cls)).float()

    np.random.seed(0)
    for ims, second_ims, third_ims, boxes, hois, verbs, objects in tqdm(test_loader):
        ims = ims.cuda(non_blocking=True)
        boxes = [[Boxes(b.cuda(non_blocking=True))] for b in boxes]
        hois = hois.cuda(non_blocking=True)
        verbs = verbs.cuda(non_blocking=True)
        objects = objects.cuda(non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                if config['eval_mode']:
                    logits_verb, logits_object = model(ims, boxes)
                    logits = utils.hico_logit_conversion_to_hoi(logits_verb, logits_object, correspondance).detach()
                else:
                    logits = model(ims, boxes)
                loss = F.binary_cross_entropy_with_logits(logits, hois.float(), pos_weight=torch.ones(logits.size(1)).to(logits)*10)
                y_score[current_idx.item():current_idx.item()+logits.size(0)] = logits.detach().cpu()
                y_true[current_idx.item():current_idx.item()+logits.size(0)] = hois.detach().cpu()
                if config['eval_mode']:
                    verb_score[current_idx.item():current_idx.item()+logits.size(0)] = logits_verb.detach().cpu()
                    obj_score[current_idx.item():current_idx.item()+logits.size(0)] = logits_object.detach().cpu()
                    verb_true[current_idx.item():current_idx.item()+logits.size(0)] = verbs.detach().cpu()
                    obj_true[current_idx.item():current_idx.item()+logits.size(0)] = objects.detach().cpu()
                    verb_true[current_idx.item():current_idx.item()+logits.size(0)] = verbs.detach().cpu()
                    obj_true[current_idx.item():current_idx.item()+logits.size(0)] = objects.detach().cpu()
                current_idx += logits.size(0)

        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = logits.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        loss_meter.update(loss.item(), logits.size(0))
    final_y_true, final_y_score = gather_score_label(dist, test_loader, current_idx.cuda(), y_true.cuda(), y_score.cuda(), args.multiprocessing_distributed)
    map = compute_map_hico(final_y_true, final_y_score)
    if config['eval_mode'] == 1:
        map_unseen = compute_map_hico(final_y_true, final_y_score, easy=True)
    elif config['eval_mode'] == 2:
        map_unseen = compute_map_hico(final_y_true, final_y_score, hard=True)
    else:
        map_unseen = 0

    map_rare = compute_map_hico(final_y_true, final_y_score, rare_only=True)
    if config['eval_mode']:
        final_verb_true, final_verb_score = gather_score_label(dist, test_loader, current_idx.cuda(), verb_true.cuda(), verb_score.cuda(), args.multiprocessing_distributed)
        final_obj_true, final_obj_score = gather_score_label(dist, test_loader, current_idx.cuda(), obj_true.cuda(), obj_score.cuda(), args.multiprocessing_distributed)
        map_verb = compute_map_hico(final_verb_true, final_verb_score)
        map_obj = compute_map_hico(final_obj_true, final_obj_score)

    if config['eval_mode']:
        return loss_meter.avg, {'mAP': map, 'mAP_unseen': map_unseen, 'mAP_rare': map_rare, 'mAP_verb': map_verb, 'mAP_obj': map_obj}
    else:
        return loss_meter.avg, {'mAP': map, 'mAP_unseen': map_unseen, 'mAP_rare': map_rare}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    parser.add_argument('--svname', default=None)
    parser.add_argument('--save_dir', default='./save_dist')
    parser.add_argument('--tag', default=None)
    # parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_model', default=None)

    # distributed training
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.multiprocessing_distributed = True

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    if args.opts is not None:
        config = utils.override_cfg_from_list(config, args.opts)
    print('config:')
    print(config)
    main(config)
