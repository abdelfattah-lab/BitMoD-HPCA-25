import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
from typing import List
CLIPMIN = 1e-5


#################################  3-bit Datatypes  #################################
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

FP3_SP_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_SP_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SR_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_SR_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

FP3_LIST = [FP3_SP_POS, FP3_SP_NEG, FP3_SR_POS, FP3_SR_NEG]

#################################  4-bit Datatypes  #################################
INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

FP4_SM_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_SM_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_SR_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_SR_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_LIST = [FP4_SM_POS, FP4_SM_NEG, FP4_SR_POS, FP4_SR_NEG]

class NonUniformRoundSTE(nn.Module):
    def __init__(self, levels: Union[torch.Tensor, list]):
        """
        Initializes the NonUniformRoundSTE module.

        Args:
            levels (Union[torch.Tensor, list]): A sorted list or tensor of quantization levels.
        """
        super(NonUniformRoundSTE, self).__init__()
        
        # Convert levels to a sorted 1D tensor
        if isinstance(levels, list):
            levels = torch.tensor(levels, dtype=torch.float32)
        elif isinstance(levels, torch.Tensor):
            levels = levels.float()
        else:
            raise TypeError("Levels must be a list or a torch.Tensor.")
        
        # Ensure the levels are sorted
        sorted_levels, _ = torch.sort(levels)
        self.register_buffer('levels', sorted_levels)
        self.min_value = float(min(levels))
        self.max_value = float(max(levels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=self.min_value, max=self.max_value)
        indices = torch.bucketize(x, self.levels, right=False)
        indices = torch.clamp(indices, min=1, max=len(self.levels)-1)
        left = self.levels[indices - 1]
        right = self.levels[indices]
        left_diff = torch.abs(x - left)
        right_diff = torch.abs(x - right)
        mask = right_diff < left_diff
        rounded = torch.where(mask, right, left)
        return (rounded - x).detach() + x
    



def select_min_score_tensor(result_list: List[torch.Tensor], 
                           score_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Selects, for each row, the tensor with the lowest score among multiple options.

    Args:
        result_list (List[torch.Tensor]): 
            A list of tensors, each of shape (M, N) with gradients.
        score_list (List[torch.Tensor]): 
            A list of score tensors, each of shape (M, 1).

    Returns:
        torch.Tensor: 
            Combined tensor of shape (M, N) selecting rows 
            from the tensor with the lowest score.
    """

    scores = torch.cat(score_list, dim=1)  # Shape: (M, K)
    # Find the index of the minimum score for each row
    min_indices = torch.argmin(scores, dim=1)  # Shape: (M,)
    # Stack all result tensors into a single tensor of shape (K, M, N)
    stacked_results = torch.stack(result_list, dim=0)  # Shape: (K, M, N)
    md = min_indices.size(0)
    row_indices = torch.arange(md, device=min_indices.device)  # Shape: (M,)
    selected = stacked_results[min_indices, row_indices, :]  # Shape: (M, N)
    
    return selected

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


class BitMoDQuantizer(nn.Module):

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,

        datatype='int',
        search_method='exhaustive',
        **kwargs
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits  


        self.datatype = datatype
        self.search_method = search_method

        # initialize the quantization metadata based on the datatype
        if datatype == 'int': # integer quantization
            self.symmetric = symmetric
            self.disable_zero_point = disable_zero_point
            if self.disable_zero_point:
                self.qmin = -(2 ** (n_bits - 1))
                self.qmax = 2 ** (n_bits - 1) - 1
            else:
                self.qmin = 0
                self.qmax = 2 ** (n_bits) - 1
            self.per_channel_axes = per_channel_axes
            self.metric = metric
            self.cluster_counts = None
            self.cluster_dim = None

            self.scale = None
            self.zero_point = None
            self.round_zero_point = None

            self.cached_xmin = None
            self.cached_xmax = None
            self.dynamic = dynamic
            self.dynamic_method = dynamic_method

        elif datatype == 'fp': # floating point quantization
            if n_bits == 3:
                fp_list = FP3
            elif n_bits == 4:
                fp_list = FP4_E2M1
            elif n_bits == 16: # just a place holder to keep code running as 16 bits wont be quantized at all
                pass
            else:
                raise NotImplementedError("bitwidth not supported by fp quantizer, only 3 and 4 are supported")
            if n_bits < 16:
                self.min_value = float(min(fp_list))
                self.max_value = float(max(fp_list))
                self.fp_range = self.max_value - self.min_value
                self.zero_ratio = - self.min_value / self.fp_range
                self.fp_rounder = NonUniformRoundSTE(levels=fp_list)

        elif datatype == 'mod': # bit mod quantization
            # initialize common part for both exhaustive and differentiable search
            if n_bits == 3:
                fp_lists = FP3_LIST
            elif n_bits == 4:
                fp_lists = FP4_LIST
            elif n_bits == 16: # just a place holder to keep code running as 16 bits wont be quantized at all
                pass
            else:
                raise NotImplementedError("bitwidth not supported by mod quantizer, only 3 and 4 are supported")
            if n_bits < 16:
                self.fp_rounders = nn.ModuleList()
                self.fp_ranges = []
                self.zero_ratios = []
                self.fp_maxs = []
                self.fp_mins = []
                for levels in fp_lists:
                    min_value = float(min(levels))
                    max_value = float(max(levels))
                    fp_range = max_value - min_value
                    self.fp_ranges.append(fp_range)
                    zero_ratio = - min_value / fp_range
                    self.zero_ratios.append(zero_ratio)
                    self.fp_rounders.append(NonUniformRoundSTE(levels))
                    self.fp_maxs.append(max_value)
                    self.fp_mins.append(min_value)
                self.fp_options_len = len(fp_lists)
                self.fp_ranges = torch.tensor(self.fp_ranges)
                self.fp_maxs = torch.tensor(self.fp_maxs)
                self.fp_mins = torch.tensor(self.fp_mins)

                # extra setup for differentiable search
                if self.search_method == 'differentiable' or self.search_method == 'diff':
                    self.temperature = None
                    if lwc:
                        if group_size:
                            dim1qc = int(shape[0]*math.ceil(shape[1]/group_size))
                        else:
                            dim1qc = shape[0]
                        # Initialize quantization choice parameters
                        self.quant_choice_param = nn.Parameter(torch.zeros((dim1qc, self.fp_options_len)))
                        
        else:
            raise NotImplementedError("datatype not supported, only int, fp, mod are supported")
        
        
        
        self.group_size = group_size

        ############### initialize LWC parameters ################
        self.lwc = lwc
        init_value = 3.0 if (datatype == 'fp' or datatype == 'mod') else 4.0      # inti value of learnable weight clipping
        self.deficiency = 0
        self.group_size = group_size
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            if datatype != 'int' and symmetric:
                self.symbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            else:
                self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
                self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()
        ###########################################################

        self.enable = True


    def forward(self, x):
        if self.n_bits >= 16 or not self.enable:
            return x
        
        if self.datatype == 'int':
            return self.forward_int(x)
        elif self.datatype == 'fp':
            return self.forward_fp(x)
        elif self.datatype == 'mod':
            if self.search_method == 'exhaustive' or self.search_method == 'exha':
                return self.forward_dyn(x)
            elif self.search_method == 'differentiable' or self.search_method == 'diff':
                return self.forward_dynl(x)
            else:
                raise NotImplementedError("search method not supported, only exhaustive and differentiable are supported")
        else:
            raise NotImplementedError("datatype not supported, only int, fp, mod and arb are supported")

# =============================================== fp quantization ===============================================

    def forward_fp(self, x):
        # pad zeros and reshape for group quantization
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)


        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin

        # xmax and min are now clipped value
        scale = (xmax - xmin) / self.fp_range        
        zero_point_raw = xmax * self.zero_ratio + xmin * (1 - self.zero_ratio)
        round_zero_point = (zero_point_raw / scale).round()

        Wq = self.fp_rounder( (x / scale) - round_zero_point)
        x_dequant = (Wq + round_zero_point) * scale

        # unpad zero and reshape for degroup 
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    
# =============================================== dyn quantization (bitmod + exhaustive) ===============================================

    def forward_dyn(self, x):
        # pad zeros and reshape for group quantization
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)


        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin

        x_dequants = [] # Weight quantized results for each option
        scores = [] # scores for each quantization option

        for i in range(self.fp_options_len):
            scale = (xmax - xmin) / self.fp_ranges[i]
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point_raw = xmax * self.zero_ratios[i] + xmin * (1 - self.zero_ratios[i])
            zero_point_int = (zero_point_raw / scale).round()
            Wq = self.fp_rounders[i]((x / scale) - zero_point_int)
            x_dequant = (Wq + zero_point_int) * scale
            x_dequants.append(x_dequant)
            with torch.no_grad():
                score = torch.sum( (x - x_dequant).pow(2), dim=1, keepdim=True)
                scores.append(score)
        # Select the best quantization option for each row
        x_best = select_min_score_tensor(x_dequants, scores)


        # unpad zero and reshape for degroup 
        if self.group_size:
            x_best = x_best.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_best = x_best[:,:-self.deficiency]
        return x_best

# =============================================== dynl quantization (bitmod + differentiable ) ===============================================

    def forward_dynl(self, x):
        G = 128 # default value, always overwrite.

        # pad zeros and reshape for group quantization
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
            G = x.shape[0]
        
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)

        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin

        x_dequants = []  # Quantized results for each option
        for i in range(self.fp_options_len):
            scale = (xmax - xmin) / self.fp_ranges[i]
            zero_point_raw = xmax * self.zero_ratios[i] + xmin * (1 - self.zero_ratios[i])
            zero_point_int = (zero_point_raw / scale).round()
            Wq = self.fp_rounders[i]((x / scale) - zero_point_int)
            x_dequant = (Wq + zero_point_int) * scale
            x_dequants.append(x_dequant)
        
        # Stack quantized outputs: shape (G, K, group_size)
        x_dequants_tensor = torch.stack(x_dequants, dim=1)  # Shape: (G, K, group_size)

        if self.training:
            logits = self.quant_choice_param  # Shape: (G, K)
            if self.temperature is not None:
                weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1)  # Shape: (G, K)
            else:
                print("Quantizer temperature missing?")
                weights = F.softmax(logits, dim=1)
            weights_expanded = weights.unsqueeze(-1)  # Shape: (G, K, 1)
            x_weighted = torch.sum(weights_expanded * x_dequants_tensor, dim=1)  # Shape: (G, group_size)
        else:
            # During evaluation, use hard selection
            indices = torch.argmax(self.quant_choice_param, dim=1)  # Shape: (G,)
            x_weighted = x_dequants_tensor[torch.arange(G), indices, :]  # Shape: (G, group_size)
        
        # Unpad zeros and reshape back
        if self.group_size:
            x_weighted = x_weighted.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_weighted = x_weighted[:, :-self.deficiency]
        return x_weighted

# ======================================== original omniquant int quantization forward =======================================

    def fake_quant_int(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    
    def forward_int(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration_int(x)
        else:
            raise NotImplementedError()   

        x_dequant = self.fake_quant_int(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration_int(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

# ===========================================================================================================================

    def register_scales_and_zeros(self):
        # create empty field in the case of fp/mod quantization
        if not hasattr(self, 'scale'):
            self.scale = None
        if not hasattr(self, 'round_zero_point'):
            self.round_zero_point = None

        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
   