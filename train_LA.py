# coding:utf-8
from __future__ import print_function
# import torch
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
# import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
import datetime
from utils import *
import cfgs.cfgs_LA as cfgs
from collections import OrderedDict
import time
import sys
import os

def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)
    # label_flatten = torch.LongTensor(label_flatten)
    label_flatten = mindspore.Tensor(label_flatten, mindspore.int64)
    # label_length = torch.IntTensor(label_length)
    label_length = mindspore.Tensor(label_length, mindspore.int32)
    return (label_flatten, label_length)

# 怎么改？
def Train_or_Eval(model, state='Train'):
    if state == 'Train':
        model.train()   # 该模式启用BN和Dropout
    else:
        model.eval()    # 该模式禁止BN和Dropout

# 怎么改？
def Zero_Grad(model):
    model.zero_grad()   # 将所有模型参数置0

# 怎么改？
def Updata_Parameters(optimizers, frozen):
    for i in range(0, len(optimizers)):
        if i not in frozen:
            optimizers[i].step()

def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    # 怎么改？mindspore没有dataloader，需要重写
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])
    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    # 怎么改？
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    return train_loader, test_loader

# 怎么改？
def load_network():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 选设备
    model_VL = cfgs.net_cfgs['VisualLAN'](**cfgs.net_cfgs['args'])
    model_VL = model_VL.to(device)  # 模型加载到设备上
    model_VL = torch.nn.DataParallel(model_VL)  # 设置并行
    if cfgs.net_cfgs['init_state_dict'] != None:    # 若有训练好的参数模型，则执行下述
        fe_state_dict_ori = torch.load(cfgs.net_cfgs['init_state_dict'])    #
        fe_state_dict = OrderedDict()
        for k, v in fe_state_dict_ori.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            fe_state_dict[k] = v
        model_dict_fe = model_VL.state_dict()   #
        state_dict_fe = {k: v for k, v in fe_state_dict.items() if k in model_dict_fe.keys()}
        model_dict_fe.update(state_dict_fe) #？
        #model_VL.load_para_into_net(model_dict_fe, fe_state_dict_ori) #
        model_VL.load_param_into_net(model_VL, model_dict_fe)
    return model_VL


def generate_optimizer(model):
    if cfgs.global_cfgs['step'] != 'LF_2':
        # out = torch.optim.Adam([{'params': model.parameters(), 'lr': cfgs.optimizer_cfgs['optimizer_0_args']['lr']}])
        out = nn.Adam([{'params': model.parameters(), 'lr': cfgs.optimizer_cfgs['optimizer_0_args']['lr']}])
        scheduler = cfgs.optimizer_cfgs['optimizer_0_scheduler'](out, **cfgs.optimizer_cfgs['optimizer_0_scheduler_args'])
        return out, scheduler
    else:
        id_mlm = id(model.module.MLM_VRM.MLM.parameters())  # id()函数实际上就是指针的意思
        # 下面这句会有问题吗？
        id_pre_mlm = id(model.module.MLM_VRM.Prediction.pp_share.parameters()) + id(model.module.MLM_VRM.Prediction.w_share.parameters())
        id_total = id_mlm + id_pre_mlm
        # out = torch.optim.Adam([{'params': filter(lambda p: id(p) == id_total, model.parameters()), 'lr': cfgs.optimizer_cfgs['optimizer_0_args']['lr']},
        #                         {'params': filter(lambda p: id(p) != id_total, model.parameters()),'lr': cfgs.optimizer_cfgs['optimizer_0_args']['lr'] * 0.1}])
        out = nn.Adam([{'params': filter(lambda p: id(p) == id_total, model.parameters()), 'lr': cfgs.optimizer_cfgs['optimizer_0_args']['lr']},
                                {'params': filter(lambda p: id(p) != id_total, model.parameters()),'lr': cfgs.optimizer_cfgs['optimizer_0_args']['lr'] * 0.1}])
        scheduler = cfgs.optimizer_cfgs['optimizer_0_scheduler'](out, **cfgs.optimizer_cfgs['optimizer_0_scheduler_args'])
        return out, scheduler

def _flatten(sources, lengths):
    # return torch.cat([t[:l] for t, l in zip(sources, lengths)])
    concat_op = ops.Concat
    return concat_op([t[:l] for t, l in zip(sources, lengths)])

# 问题较多
def test(test_loader, model, tools, best_acc):
    Train_or_Eval(model, 'Eval')    # pytorch这个train和eval模式在mindspore怎么调？
    for sample_batched in test_loader:  #testloader没有了，还能for循环吗？是什么样的，需要试验一下
        data = sample_batched['image']
        label = sample_batched['label']
        target = tools[0].encode(label)
        #data = data.cuda()
        target = target
        label_flatten, length = tools[1](target)
        #target, label_flatten = target.cuda(), label_flatten.cuda()
        output, out_length = model(data, target, '', False)
        tools[2].add_iter(output, out_length, length, label)
    best_acc, change = tools[2].show_test(best_acc)
    Train_or_Eval(model, 'Train')
    return best_acc, change

if __name__ == '__main__':
    model = load_network()  # 注意到函数里去看
    optimizer, optimizer_scheduler = generate_optimizer(model)
    # criterion_CE = nn.CrossEntropyLoss().cuda()   # .cuda是为了将Tenosr 拷贝到 cuda 内存，而mindspore初始device设置好后，网络和tensor都自动拷贝到device上
    criterion_CE = nn.SoftmaxCrossEntropyWithLogits()
    # L1_loss = nn.L1Loss().cuda()
    L1_loss = nn.L1Loss()

    train_loader, test_loader = load_dataset()  # 有点问题

    # tools prepare
    # Attention_AR_counter是个啥玩意，怎么来的，好像没见函数定义哇
    # 还有那个cha_encdec
    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                             cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_rem = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_sub = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])

    # train
    total_iters = len(train_loader) # 注意有loader
    loss_show = 0 
    time_cal = 0
    ratio_res = 0.5
    ratio_sub = 0.5
    best_acc = 0
    loss_ori_show = 0
    loss_mas_show = 0
    if not os.path.isdir(cfgs.saving_cfgs['saving_path']):
        os.mkdir(cfgs.saving_cfgs['saving_path'])
    for nEpoch in range(0, cfgs.global_cfgs['epoch']):
        for batch_idx, sample_batched in enumerate(train_loader):   # 注意有loader
            # data_prepare
            data = sample_batched['image']
            label = sample_batched['label']  # original string
            label_res = sample_batched['label_res']  # remaining string
            label_sub = sample_batched['label_sub']  # occluded character
            label_id = sample_batched['label_id']  # character index
            target = encdec.encode(label)
            target_res = encdec.encode(label_res)
            target_sub = encdec.encode(label_sub)
            Train_or_Eval(model, 'Train')
            # 后续的.cuda是不是都可以省略
            #data = data.cuda()
            label_flatten, length = flatten_label(target)
            label_flatten_res, length_res = flatten_label(target_res)
            label_flatten_sub, length_sub = flatten_label(target_sub)
            #target, label_flatten, target_res, target_sub, label_flatten_res = target.cuda(), label_flatten.cuda(), target_res.cuda(), target_sub.cuda(), label_flatten_res.cuda()
            #label_flatten_sub, label_id = label_flatten_sub.cuda(), label_id.cuda()
            # prediction
            text_pre, text_rem, text_mas, att_mask_sub = model(data, label_id, cfgs.global_cfgs['step'])
            # loss_calculation
            if cfgs.global_cfgs['step'] == 'LF_1':
                text_pre = _flatten(text_pre, length)
                pre_ori, label_ori = train_acc_counter.add_iter(text_pre, length.long(), length, label)

                loss_ori = criterion_CE(text_pre, label_flatten)
                loss = loss_ori
            else:
                text_pre = _flatten(text_pre, length)
                text_rem = _flatten(text_rem, length_res)
                text_mas = _flatten(text_mas, length_sub)
                pre_ori, label_ori = train_acc_counter.add_iter(text_pre, length.long(), length, label)
                pre_rem, label_rem = train_acc_counter_rem.add_iter(text_rem, length_res.long(), length_res, label_res)
                pre_sub, label_sub = train_acc_counter_sub.add_iter(text_mas, length_sub.long(), length_sub, label_sub)

                loss_ori = criterion_CE(text_pre, label_flatten)
                loss_res = criterion_CE(text_rem, label_flatten_res)
                loss_mas = criterion_CE(text_mas, label_flatten_sub)
                loss = loss_ori + loss_res * ratio_res + loss_mas * ratio_sub
                loss_ori_show += loss_res
                loss_mas_show += loss_mas
            # loss for display
            loss_show += loss
            # optimize
            Zero_Grad(model)
            #loss.backward()
            # 梯度裁剪，防止梯度爆炸问题，三个参数分别为：parameters: 网络参数，max_norm: 该组网络参数梯度的范数上线，norm_type: 范数类型
            # nn.utils.clip_grad_norm_(model.parameters(), 20, 2) 
            # 这玩意mindspore好像没有
            optimizer.step()
            # display
            if batch_idx % cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
                loss_show = loss_show / cfgs.global_cfgs['show_interval']
                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print(
                    'Epoch: {}, Iter: {}/{}, Loss VisionLAN: {:0.4f}'.format(
                        nEpoch,
                        batch_idx,
                        total_iters,
                        loss_show))
                loss_show = 0
                train_acc_counter.show()
                if cfgs.global_cfgs['step'] != 'LF_1':
                    print(
                        'orignial: {}, mask_character pre/gt: {}/{}, other pre/gts: {}/{}'.format(
                            label[0],
                            pre_sub[0],
                            label_sub[0],
                            pre_rem[0],
                            label_rem[0]))
                    loss_mas_show = loss_mas_show / cfgs.global_cfgs['show_interval']
                    loss_ori_show = loss_ori_show / cfgs.global_cfgs['show_interval']
                    print('loss for mas/rem: {}/{}'.format(loss_mas_show,loss_ori_show))
                    loss_ori_show = 0
                    loss_mas_show = 0
            sys.stdout.flush()
            # evaluation during training
            if batch_idx % cfgs.global_cfgs['test_interval'] == 0 and batch_idx != 0:
                print('Testing during training:')
                best_acc, if_save = test((test_loader),
                                        model,
                                        [encdec,
                                         flatten_label,
                                         test_acc_counter], best_acc)
                if if_save:
                        # torch.save(model.state_dict(),
                        #            cfgs.saving_cfgs['saving_path'] + 'best_acc_M.pth')
                        mindspore.save_checkpoint(model.state_dict(),
                                    cfgs.saving_cfgs['saving_path'] + 'best_acc_M.pth')
        # save each epoch
        if nEpoch % cfgs.saving_cfgs['saving_epoch_interval'] == 0:
            # torch.save(model.state_dict(),
            #            cfgs.saving_cfgs['saving_path'] + 'E{}.pth'.format(
            #                nEpoch))
            mindspore.save_checkpoint(model.state_dict(),
                       cfgs.saving_cfgs['saving_path'] + 'E{}.pth'.format(
                           nEpoch))         
        optimizer_scheduler.step()
