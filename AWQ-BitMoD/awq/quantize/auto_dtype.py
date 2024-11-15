import torch
import torch.nn as nn
from typing import Optional
import time


#################################  3-bit Datatypes  #################################
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_ER_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_ER_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_EA_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_EA_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

#################################  4-bit Datatypes  #################################
INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_ER_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_ER_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_EA_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_EA_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


DATATYPE_MAPPING_3_BIT = {
    'int3': INT3, 'fp3': FP3, 
    'fp3_er_pos': FP3_ER_POS, 'fp3_er_neg': FP3_ER_NEG, 
    'fp3_ea_pos': FP3_EA_POS, 'fp3_ea_neg': FP3_EA_NEG, 
}

DATATYPE_MAPPING_4_BIT = {
    'int4': INT4, 'fp4': FP4_E2M1, 
    'fp4_er_pos': FP4_ER_POS, 'fp4_er_neg': FP4_ER_NEG, 
    'fp4_ea_pos': FP4_EA_POS, 'fp4_ea_neg': FP4_EA_NEG, 
}


@torch.no_grad()
def quant_dtype(w_fp16, wq_bits:int=4, datatype: str=""):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    else:
        raise ValueError(f"Currently only support 3-bit and 4-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    rmax = torch.amax(w_fp16.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    x = w_fp16 / scale_fp

    shape = x.shape
    xhard = x.view(-1)
    value_s = torch.tensor(allow_value).type_as(x)
    idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).pow(2).min(dim=0)[1]
    q_tensor = value_s[idxs].view(shape)
    w_fp16 = q_tensor * scale_fp 

    return w_fp16, scale_fp


@torch.no_grad()
def search_group_dtype(w_fp16, wq_bits:int=4):
    if wq_bits == 3:
        datatype_list = ['fp3_er_pos', 'fp3_er_neg', 'fp3_ea_pos', 'fp3_ea_neg']
    elif wq_bits == 4:
        datatype_list = ['fp4_er_pos', 'fp4_er_neg', 'fp4_ea_pos', 'fp4_ea_neg']
    else:
        raise ValueError(f"Currently only support 3-bit and 4-bit BitMoD quantization, not {wq_bits}-bit")

    K, _ = w_fp16.size() # output channel, input channel
    w_dequant = torch.zeros_like(w_fp16)
    scale = torch.zeros([K, 1], dtype=torch.float16, device=w_fp16.device)

    error = torch.full([K], 1e3, dtype=torch.float16, device=w_fp16.device)
    for datatype in datatype_list:
        w_fp16_tmp, scale_tmp = quant_dtype(w_fp16, wq_bits=wq_bits, datatype=datatype)
        quant_error = (w_fp16_tmp - w_fp16).abs().pow(2).mean(-1)
        mask_update = torch.lt(quant_error, error)
        error[mask_update] = quant_error[mask_update]
        w_dequant[mask_update] = w_fp16_tmp[mask_update]
        scale[mask_update] = scale_tmp[mask_update]

        del w_fp16_tmp, scale_tmp
        del quant_error
        del mask_update
    
    return w_dequant, scale
