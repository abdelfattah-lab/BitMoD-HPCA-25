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
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_ER_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_ER_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_EA_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_EA_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

#################################  5-bit Datatypes  #################################
INT5 = [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FLINT5 = [-64.0, -32.0, -24.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 32.0, 64.0]
FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP5_E3M1 = [-192.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0, -16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]

#################################  6-bit Datatypes  #################################
INT6 = [
    -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
]
FP6_E2M3 = [
    -60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0
]
FP6_E3M2 = [
    -448.0, -384.0, -320.0, -256.0, -224.0, -192.0, -160.0, -128.0, -112.0, -96.0, -80.0, -64.0, -56.0, -48.0, -40.0, -32.0, -28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0, 320.0, 384.0, 448.0
]

DATATYPE_MAPPING_3_BIT = {
    'int3': INT3, 'fp3': FP3, 
    'fp3_er_pos': FP3_ER_POS, 'fp3_er_neg': FP3_ER_NEG, 
    'fp3_ea_pos': FP3_EA_POS, 'fp3_ea_neg': FP3_EA_NEG, 
}
DATATYPE_MAPPING_3_BIT_MX = {
    'mx_int3': INT3, 'mx_fp3': FP3
}

DATATYPE_MAPPING_4_BIT = {
    'int4': INT4, 'fp4': FP4_E2M1, 'flint4': FLINT4,
    'fp4_er_pos': FP4_ER_POS, 'fp4_er_neg': FP4_ER_NEG, 
    'fp4_ea_pos': FP4_EA_POS, 'fp4_ea_neg': FP4_EA_NEG, 
}
DATATYPE_MAPPING_4_BIT_MX = {
    'mx_int4': INT4, 'mx_fp4': FP4_E2M1 
}

DATATYPE_MAPPING_5_BIT = {
    'int5': INT5, 'fp5': FP5_E2M2, 'flint5': FLINT5,
    'fp5_e2m2': FP5_E2M2, 'fp5_e3m1': FP5_E3M1
}

DATATYPE_MAPPING_6_BIT = {
    'int6': INT6, 'fp6': FP6_E2M3, 
    'fp6_e2m3': FP6_E2M3, 'fp6_e3m2': FP6_E3M2
}


@torch.no_grad()
def quant_int(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Symmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)

    w_fp16_new = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_int_asym(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Asymmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**wq_bits - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    zeropoint = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=qmin, max=qmax)

    w_fp16_new = (q_tensor - zeropoint) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_mx(w_fp16, wq_bits:int=4, datatype: str="", group_size: int=32):
    """
        MX quantization.
        Reference: https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py
    """ 
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_MX
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_MX
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size() # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
    
    shared_exp, _ = torch.max(w_fp16_new.abs(), dim=-1, keepdim=True)
    shared_exp = torch.floor(torch.log2(shared_exp))
    w_fp16_new = w_fp16_new / (2**shared_exp)
    qmax = max([abs(x) for x in allow_value])
    scale = 1 / (qmax / 2)
    x = w_fp16_new / scale

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale * (2**shared_exp)
    return w_fp16_new.reshape(K, C).to(torch.float16)


@torch.no_grad()
def quant_datatype(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    else:
        raise ValueError(f"Currently only support 3-, 4-, 5-, and 6-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)

    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    x = w_fp16_new / scale_fp

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale_fp 

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def search_datatype(w_fp16, wq_bits:int=4, datatype: str='mixed_bitmod', group_size: Optional[int]=None):
    if wq_bits == 3:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg', 'fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int3', 'fp3']
    elif wq_bits == 4:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg', 'fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int4', 'flint4']
    else:
        raise ValueError(f"Currently only support 3-bit and 4-bit mixed quantization, not {wq_bits}-bit")

    K, C = w_fp16.size() # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)
    
    error = torch.full([K, NUM_GROUP], 1e3, dtype=w_fp16.dtype, device=w_fp16.device)
    for datatype in datatype_list:
        w_fp16_tmp = quant_datatype(w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None)
        quant_error = (w_fp16_tmp - w_fp16).pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_tmp[update_mask]

        del w_fp16_tmp, quant_error, update_mask
    
    return q_tensor.reshape(K, C)


def quant_model(model, wq_bits: Optional[int]=None, wq_datatype: Optional[str]=None, wq_groupsize: Optional[int]=None):
    if (wq_datatype is None) or (wq_datatype in ["fp16", "fp32"]):
        print("Not applying quantization")
        time.sleep(2)
    elif (wq_datatype.startswith("int")) and ("asym" in wq_datatype):
        print(f"Applying asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int_asym(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif (wq_datatype.startswith("int")) and ("asym" not in wq_datatype):
        print(f"Applying symmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif ("mx" in wq_datatype):
        '''
            We use hard-coded group size 32 based on the Open Compute Standard
            https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
        '''
        print(f"Applying MX quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: 32")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_mx(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=32)
    else:
        print(f"Applying mixed datatype quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = search_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)