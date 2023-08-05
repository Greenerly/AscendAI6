import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules.modules import Transforme_Encoder, Prediction, Transforme_Encoder_light
import torchvision
import modules.resnet as resnet

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class MLM(nn.Module):
    '''
    Architecture of MLM
    '''
    def __init__(self, n_dim=512):
        super(MLM, self).__init__()
        self.MLM_SequenceModeling_mask = Transforme_Encoder(n_layers=2, n_position=256)     #函数内部还未修改
        self.MLM_SequenceModeling_WCL = Transforme_Encoder(n_layers=1, n_position=256)      #函数内部还未修改
        self.pos_embedding = nn.Embedding(25, 512)  # 一致
        self.w0_linear = nn.Dense(1, 256)   #一致
        self.wv = nn.Dense(n_dim, n_dim)    #一致
        self.active = nn.Tanh()             #一致
        self.we = nn.Dense(n_dim, 1)
        self.sigmoid = nn.Sigmoid()         #一致

    def forward(self, input, label_pos, state=False):
        # transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(input, src_mask=None)[0]         #函数调用 未修改
        # position embedding layer
        pos_emb = self.pos_embedding(label_pos.long())          #函数调用，修改完成

        #pos_emb = self.w0_linear(torch.unsqueeze(pos_emb, dim=2)).permute(0, 2, 1)      #函数调用，修改完成torch.unsqueeze->mindspore.ops.ExpandDims permute->Transpose
        output_tmp = ops.ExpandDims(pos_emb, dim=2)
        pos_emb = self.w0_linear(ops.Transpose(output_tmp, (0, 2, 1)))        #permute和transpose函数接口略有不同，transpose要输入参数

        # fusion position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))                     #修改完成
        att_map_sub = self.we(att_map_sub)  # b,256,1                                   #修改完成
        #att_map_sub = self.sigmoid(att_map_sub.permute(0, 2, 1))  # b,1,256             #修改完成 permute注意
        att_map_sub = self.sigmoid(ops.Transpose(att_map_sub, (0, 2, 1)))

        # WCL
        ## generate inputs for WCL
        #f_res = input * (1 - att_map_sub.permute(0, 2, 1)) # second path with remaining string      #修改完成 permute注意
        f_res = input * (1- ops.Transpose(att_map_sub, (0, 2, 1)))
        #f_sub = input * (att_map_sub.permute(0, 2, 1)) # first path with occluded character 
        f_sub = input * (ops.Transpose(att_map_sub, (0, 2, 1)))
        ## transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)[0]                  #函数内部调用 未修改
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)[0]                  #函数内部调用 未修改
        return f_res, f_sub, att_map_sub

def trans_1d_2d(x):
    b, w_h, c = x.shape  # b, 256, 512
    #x = x.permute(0, 2, 1)
    x = ops.Transpose(x, (0, 2, 1))         #修改完成

    #x = x.view(b, c, 32, 8)    
    x = x.view((b, c, 32, 8))               #修改完成

    #x = x.permute(0, 1, 3, 2)  # [16, 512, 8, 32]
    x = ops.Transpose(x, (0, 1, 3, 2))      #修改完成
    return x

class MLM_VRM(nn.Module):
    '''
    MLM+VRM, MLM is only used in training.
    ratio controls the occluded number in a batch.
    The pipeline of VisionLAN in testing is very concise with only a backbone + sequence modeling(transformer unit) + prediction layer(pp layer).
    input: input image
    label_pos: character index
    training_stp: LF or LA process
    output
    text_pre: prediction of VRM
    test_rem: prediction of remaining string in MLM
    text_mas: prediction of occluded character in MLM
    mask_c_show: visualization of Mask_c
    '''
    def __init__(self,):
        super(MLM_VRM, self).__init__()
        self.MLM = MLM()            # 获取MLM模型
        self.SequenceModeling = Transforme_Encoder(n_layers=3, n_position=256)      # 内部函数调用，未修改
        self.Prediction = Prediction(n_position=256, N_max_character=26, n_class=37) # N_max_character = 1 eos + 25 characters   内部函数调用，未修改
        self.nclass = 37
    def forward(self, input, label_pos, training_stp, is_Train = False):
        b, c, h, w = input.shape
        nT = 25
        #input = input.permute(0, 1, 3, 2)
        input = ops.Transpose(input, (0, 1, 3, 2))
        #input = input.contiguous().view(b, c, -1)       #可以不用管contiguous，逻辑上没有意义，mindspore可能直接支持
        input = input.view((b, c, -1))          
        #input = input.permute(0, 2, 1)                 #修改完成
        input = ops.Transpose(input, (0, 2, 1))
        if is_Train:
            if training_stp == 'LF_1':
                f_res = 0
                f_sub = 0
                input = self.SequenceModeling(input, src_mask=None)[0]              #函数内部调用，可以不管
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, Train_is=True, use_mlm=False)
                return text_pre, text_pre, text_pre, text_pre
            elif training_stp == 'LF_2':
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos, state=True)
                input = self.SequenceModeling(input, src_mask=None)[0]
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, Train_is=True)
                #mask_c_show = trans_1d_2d(mask_c.permute(0, 2, 1))
                mask_c_show = trans_1d_2d(ops.Transpose(mask_c, (0, 2, 1)))              #修改完成
                return text_pre, test_rem, text_mas, mask_c_show
            elif training_stp == 'LA':
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos, state=True)
                ## use the mask_c (1 for occluded character and 0 for remaining characters) to occlude input
                ## ratio controls the occluded number in a batch
                ratio = 2
                #character_mask = torch.zeros_like(mask_c)
                character_mask = ops.ZerosLike(mask_c)          #修改完成
                character_mask[0:b // ratio, :, :] = mask_c[0:b // ratio, :, :]
                #input = input * (1 - character_mask.permute(0, 2, 1))       
                input = input * (1 - ops.Transpose(character_mask, (0, 2, 1)))       #修改完成
                # VRM
                ## transformer unit for VRM
                input = self.SequenceModeling(input, src_mask=None)[0]
                ## prediction layer for MLM and VSR
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, Train_is=True)
                #mask_c_show = trans_1d_2d(mask_c.permute(0, 2, 1))
                mask_c_show = trans_1d_2d(ops.Transpose(mask_c, (0, 2, 1)))         #修改完成
                return text_pre, test_rem, text_mas, mask_c_show
        else: # VRM is only used in the testing stage
            f_res = 0
            f_sub = 0
            contextual_feature = self.SequenceModeling(input, src_mask=None)[0]
            C = self.Prediction(contextual_feature, f_res, f_sub, Train_is=False, use_mlm=False)
            #C = C.permute(1, 0, 2)  # (25, b, 38))
            C = ops.Transpose(C, (1, 0, 2))     #修改完成
            lenText = nT
            nsteps = nT
            #out_res = torch.zeros(lenText, b, self.nclass).type_as(input.data)
            out_res = ops.Zeros(lenText, b, self.nclass)          #！！！！疑难杂症type_as

            #out_length = torch.zeros(b).type_as(input.data)             #！！！！疑难杂症 可能有问题
            out_length = ops.Zeros(b)
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                tmp_result = C[now_step, :, :]
                out_res[now_step] = tmp_result
                #tmp_result = tmp_result.topk(1)[1].squeeze(dim=1)
                tmp_topk_res = ops.TopK(sorted=True)(tmp_result, 1)[1]
                tmp_result = tmp_topk_res.squeeze(dim=1)        #已修改 可能有问题 torch的这个函数有dim参数可以选择具体删除
                for j in range(b):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            #output = torch.zeros(int(out_length.sum()), self.nclass).type_as(input.data)        #！！！！疑难杂症
            output = ops.Zeros(int(out_length.sum()), self.nclass)        #！！！！疑难杂症
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start: start + cur_length] = out_res[0: cur_length, i, :]
                start += cur_length

            return output, out_length


class VisionLAN(nn.Module):
    '''
    Architecture of VisionLAN
    input
    input: input image
    label_pos: character index
    output
    text_pre: word-level prediction from VRM
    test_rem: remaining string prediction from MLM
    text_mas: occluded character prediction from MLM
    '''
    def __init__(self, strides, input_shape):
        super(VisionLAN, self).__init__()       #super在python中是一个类，
        self.backbone = resnet.resnet45(strides, compress_layer=False)      #获取backbone module
        self.input_shape = input_shape              #获取input shape
        self.MLM_VRM = MLM_VRM()            #获取MLM_VRM模型
    def forward(self, input, label_pos, training_stp, Train_in = True):
        # extract features
        features = self.backbone(input)
        # MLM + VRM
        if Train_in:
            text_pre, test_rem, text_mas, mask_map = self.MLM_VRM(features[-1], label_pos, training_stp, is_Train=Train_in)
            return text_pre, test_rem, text_mas, mask_map
        else:
            output, out_length = self.MLM_VRM(features[-1], label_pos, training_stp, is_Train=Train_in)
            return output, out_length
