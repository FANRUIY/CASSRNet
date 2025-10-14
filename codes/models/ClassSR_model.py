import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import (
    CharbonnierLoss,
    class_loss_3class,
    average_loss_3class,
    BalanceLoss,
    EntropyLoss
)
import cv2
import numpy as np
from utils import util

logger = logging.getLogger('base')


class ClassSR_Model(BaseModel):
    def __init__(self, opt):
        super(ClassSR_Model, self).__init__(opt)

        self.scale = int(opt["scale"])
        self.name = opt['name']
        self.which_model = opt['network_G']['which_model_G']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt['train']

        # define network
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            # loss 权重
            self.l1w = float(opt["train"]["l1w"])
            self.class_loss_w = float(opt["train"]["class_loss_w"])
            self.average_loss_w = float(opt["train"]["average_loss_w"])
            self.balance_loss_w = float(opt["train"]["balance_loss_w"])
            self.entropy_loss_w = float(opt["train"]["entropy_loss_w"])

            self.pf = opt['logger']['print_freq']
            self.batch_size = int(opt['datasets']['train']['batch_size'])
            self.netG.train()

            # loss 定义
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'ClassSR_loss':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.class_loss = class_loss_3class().to(self.device)
                self.average_loss = average_loss_3class().to(self.device)
                self.balance_loss = BalanceLoss().to(self.device)
                self.entropy_loss = EntropyLoss().to(self.device)
            else:
                raise NotImplementedError(f'Loss type [{loss_type}] is not recognized.')

            # optimizer
            wd_G = train_opt.get('weight_decay_G', 0)
            optim_params = []
            if opt['fix_SR_module']:
                for k, v in self.netG.named_parameters():
                    if v.requires_grad and "class" not in k:
                        v.requires_grad = False

            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(f'Params [{k}] will not optimize.')

            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt['lr_G'],
                weight_decay=wd_G,
                betas=(train_opt['beta1'], train_opt['beta2'])
            )
            self.optimizers.append(self.optimizer_G)

            # scheduler
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer, train_opt['lr_steps'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights'],
                            gamma=train_opt['lr_gamma'],
                            clear_state=train_opt['clear_state']
                        )
                    )
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']
                        )
                    )
            else:
                raise NotImplementedError('Only MultiStepLR and CosineAnnealingLR are supported.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)
        self.LQ_path = data['LQ_path'][0]
        if need_GT:
            self.real_H = data['GT'].to(self.device)
            self.GT_path = data['GT_path'][0]

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward
        self.fake_H, self.gate_probs = self.netG(self.var_L, self.is_train)

        # loss
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        class_loss = self.class_loss(self.gate_probs)
        average_loss = self.average_loss(self.gate_probs)
        balance_loss = self.balance_loss(self.gate_probs)
        entropy_loss = self.entropy_loss(self.gate_probs)

        loss = (
            self.l1w * l_pix
            + self.class_loss_w * class_loss
            + self.average_loss_w * average_loss
            + self.balance_loss_w * balance_loss
            + self.entropy_loss_w * entropy_loss
        )

        if step % self.pf == 0:
            self.print_res(self.gate_probs)

        loss.backward()
        self.optimizer_G.step()

        # log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['class_loss'] = class_loss.item()
        self.log_dict['average_loss'] = average_loss.item()
        self.log_dict['balance_loss'] = balance_loss.item()
        self.log_dict['entropy_loss'] = entropy_loss.item()
        self.log_dict['loss'] = loss.item()

    def test(self):
        self.netG.eval()

        self.var_L = cv2.imread(self.LQ_path, cv2.IMREAD_UNCHANGED)
        self.real_H = cv2.imread(self.GT_path, cv2.IMREAD_UNCHANGED)

        if self.which_model == 'classSR_3class_rcan':
            img = self.var_L.astype(np.float32)
        else:
            img = self.var_L.astype(np.float32) / 255.

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()[None, ...].to(self.device)

        with torch.no_grad():
            srt, type = self.netG(img, False)

        if self.which_model == 'classSR_3class_rcan':
            sr_img = util.tensor2img(srt.detach()[0].float().cpu(), out_type=np.uint8, min_max=(0, 255))
        else:
            sr_img = util.tensor2img(srt.detach()[0].float().cpu())

        self.fake_H = sr_img
        self.real_H = self.real_H[0:sr_img.shape[0], 0:sr_img.shape[1], :]

        psnr = util.calculate_psnr(sr_img, self.real_H)
        flag = torch.max(type, 1)[1].data
        if flag.dim() == 0:
            flag = flag.unsqueeze(0)

        psnr_type1 = psnr_type2 = psnr_type3 = 0
        if flag == 0:
            psnr_type1 += psnr
        elif flag == 1:
            psnr_type2 += psnr
        elif flag == 2:
            psnr_type3 += psnr

        self.num_res = self.print_res(type)
        self.psnr_res = [psnr_type1, psnr_type2, psnr_type3]

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L
        out_dict['rlt'] = self.fake_H
        out_dict['num_res'] = self.num_res
        out_dict['psnr_res'] = self.psnr_res
        if need_GT:
            out_dict['GT'] = self.real_H
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, (nn.DataParallel, DistributedDataParallel)):
            net_struc_str = f'{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}'
        else:
            net_struc_str = f'{self.netG.__class__.__name__}'
        if self.rank <= 0:
            logger.info(f'Network G structure: {net_struc_str}, with parameters: {n:,d}')
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_classifier = self.opt['path']['pretrain_model_classifier']
        load_path_G_branch3 = self.opt['path']['pretrain_model_G_branch3']
        load_path_G_branch2 = self.opt['path']['pretrain_model_G_branch2']
        load_path_G_branch1 = self.opt['path']['pretrain_model_G_branch1']
        load_path_Gs = [load_path_G_branch1, load_path_G_branch2, load_path_G_branch3]

        if load_path_G is not None:
            logger.info(f'Loading model for G [{load_path_G}] ...')
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        if load_path_classifier is not None:
            logger.info(f'Loading model for classifier [{load_path_classifier}] ...')
            self.load_network_classifier_rcan(load_path_classifier, self.netG, self.opt['path']['strict_load'])
        if all([load_path_G_branch1, load_path_G_branch2, load_path_G_branch3]):
            logger.info(f'Loading model for branch1 [{load_path_G_branch1}] ...')
            logger.info(f'Loading model for branch2 [{load_path_G_branch2}] ...')
            logger.info(f'Loading model for branch3 [{load_path_G_branch3}] ...')
            self.load_network_classSR_3class(load_path_Gs, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def print_res(self, probs_or_type):
        """支持训练时(gate_probs)和测试时(type)"""
        if probs_or_type.dim() == 2 and probs_or_type.size(1) > 1:  
            # gate_probs
            flag = torch.argmax(probs_or_type, dim=1)
        else:  
            # type (one-hot)
            flag = torch.max(probs_or_type, 1)[1].data
        if flag.dim() == 0:
            flag = flag.unsqueeze(0)

        type1 = type2 = type3 = 0
        num_res = torch.zeros(3)
        for i in range(flag.size(0)):
            if flag[i] == 0:
                type1 += 1
                num_res[0] += 1
            elif flag[i] == 1:
                type2 += 1
                num_res[1] += 1
            elif flag[i] == 2:
                type3 += 1
                num_res[2] += 1
        print(f'type1:{type1} type2:{type2} type3:{type3}')
        return num_res
