import ninetoothed
import ninetoothed.language as ntl
import functools
import torch
import typing
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.custom_op import CustomOp
import torch.nn.functional as F
from vllm.logger import init_logger
from triton.language.extra import libdevice

logger = init_logger(__name__)

class SiluAndMul:
    def arrangement(x, y, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        y_arranged = y.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, y_arranged, output_arranged

    def application(x, y, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        y_fp32 = ntl.cast(y, ntl.float32)
        out = x_fp32 * (1 / (1 + ntl.exp(-x_fp32))) * y_fp32

    def premake(ndim: int):
        kernel = ninetoothed.make(
            SiluAndMul.arrangement,
            SiluAndMul.application,
            (
                ninetoothed.Tensor(ndim),
                ninetoothed.Tensor(ndim),
                ninetoothed.Tensor(ndim),
            ),
        )
        return kernel
    

kernel = {}
kernel["siluAndMul"] = {i: SiluAndMul.premake(i) for i in range(2, 4)}


def siluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x_in = x[..., :d]
    y_in = x[..., d:]
    kernel["siluAndMul"][ndim](x_in, y_in, out, BLOCK_SIZE=d)
    return out


def fake_siluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_silu_and_mul", siluAndMul, fake_impl=fake_siluAndMul)


def silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT SILU AND MUL is enabled.\033[0m")
    return torch.ops.vllm.nt_silu_and_mul(x)


class FatreluAndMul:
    def arrangement(x, y, out,threshold):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        y_arranged = y.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        out_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, y_arranged, out_arranged, threshold

    def application(x, y, out, threshold):
        x_fp32 = ntl.cast(x, ntl.float32)
        y_fp32 = ntl.cast(y, ntl.float32)

        # FATReLU: keep x if > threshold, else 0
        activated = ntl.where(x_fp32 > threshold, x_fp32,0.0)
        # Multiply
        out = activated * y_fp32

       
    def premake(ndim: int):
        kernel = ninetoothed.make(
            FatreluAndMul.arrangement,
            FatreluAndMul.application,
            (
                ninetoothed.Tensor(ndim),  # x
                ninetoothed.Tensor(ndim),  # y
                ninetoothed.Tensor(ndim),  # out
                ninetoothed.Tensor(0), # threshold
            ),
        )
        return kernel
    


kernel["fatreluAndMul"] = {
    i: FatreluAndMul.premake(i) for i in range(2, 4)
}


def fatreluAndMul(x: torch.Tensor,threshold:float) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x_in = x[..., :d]
    y_in = x[..., d:]
    kernel["fatreluAndMul"][ndim](x_in, y_in, out, threshold,BLOCK_SIZE=d)
    return out

def fake_fatreluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_fatrelu_and_mul", fatreluAndMul, fake_impl=fake_fatreluAndMul)

def fatrelu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT FATRELU AND MUL is enabled.\033[0m")
    return torch.ops.vllm.nt_fatrelu_and_mul(x)


    
class MulAndSilu:
    def arrangement(x, y, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        y_arranged = y.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, y_arranged, output_arranged

    def application(x, y, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        y_fp32 = ntl.cast(y, ntl.float32)
        out = x_fp32 * (y_fp32 / (1 + ntl.exp(-y_fp32)))
        

    def premake(ndim: int):
        kernel = ninetoothed.make(
            MulAndSilu.arrangement,
            MulAndSilu.application,
            (
                ninetoothed.Tensor(ndim),
                ninetoothed.Tensor(ndim),
                ninetoothed.Tensor(ndim),
            ),
        )
        return kernel
    

kernel["mulAndSilu"] = {i: MulAndSilu.premake(i) for i in range(2, 4)}


def mulAndSilu(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x_in = x[..., :d]   # left part: used as multiplier
    y_in = x[..., d:]   # right part: goes through SiLU
    kernel["mulAndSilu"][ndim](x_in, y_in, out, BLOCK_SIZE=d)
    return out

def fake_mulAndSilu(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_mul_and_silu", mulAndSilu, fake_impl=fake_mulAndSilu)

def mul_and_silu_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT MUL AND SILU is enabled.\033[0m")
    return torch.ops.vllm.nt_mul_and_silu(x)



#GeluAndMul部分

class GeluAndMul:
    
    def arrangement(x, y, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        y_arranged = y.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, y_arranged, output_arranged

    def application(x, y, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        y_fp32 = ntl.cast(y, ntl.float32)
        
        sqrt_2_over_pi = ntl.cast(0.7978845608028654, ntl.float32)
        c = ntl.cast(0.044715, ntl.float32)
    
        z = sqrt_2_over_pi * (x_fp32 + c * x_fp32 * x_fp32 * x_fp32)
        tanh_z = libdevice.tanh(z)
        gelu_x = 0.5 * x_fp32 * (1 + tanh_z)
        out = gelu_x * y_fp32

    def premake(ndim: int):
        kernel= ninetoothed.make(
            GeluAndMul.arrangement,
            GeluAndMul.application,
            (
                ninetoothed.Tensor(ndim),  # x (gate)
                ninetoothed.Tensor(ndim),  # y (up)
                ninetoothed.Tensor(ndim),  # out
            ),
        )
        return kernel
    


kernel["geluAndMul"] = {
        i: GeluAndMul.premake(i) for i in range(2, 4)
    }


def geluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x_in = x[..., :d]   # gate: goes through GELU
    y_in = x[..., d:]   # up: multiplier
    kernel["geluAndMul"][ndim](x_in, y_in, out, BLOCK_SIZE=d)
    return out

def fake_geluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_gelu_and_mul", geluAndMul, fake_impl=fake_geluAndMul)


def gelu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT GELU AND MUL is enabled.\033[0m")
    return torch.ops.vllm.nt_gelu_and_mul(x)



class SwigluOAIAndMul:
    def arrangement(x, y, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        y_arranged = y.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, y_arranged, output_arranged
       
       

    def application(x, y, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        y_fp32 = ntl.cast(y, ntl.float32)

        # x_fp32 = ntl.clamp(x_fp32, min=None, max=7.0) #<--min=None报错

        x_fp32 = ntl.minimum(x_fp32, 7.0)
        y_fp32 = ntl.clamp(y_fp32, min=-7.0, max=7.0)

        glu = x_fp32 * ntl.sigmoid(x_fp32 * 1.702)
        out = (y_fp32 + 1) * glu

    def premake(ndim: int):
        kernel = ninetoothed.make(
            SwigluOAIAndMul.arrangement,
            SwigluOAIAndMul.application,
            (
                ninetoothed.Tensor(ndim), # x
                ninetoothed.Tensor(ndim), # y
                ninetoothed.Tensor(ndim), # out
            ),
        )
        return kernel


kernel["swigluOAIAndMul"] = {
        i: SwigluOAIAndMul.premake(i) for i in range(2, 4)
    }

def swigluOAIAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x_in = x[..., ::2]
    y_in = x[..., 1::2]
    kernel["swigluOAIAndMul"][ndim](x_in, y_in, out, BLOCK_SIZE=d)
    return out

def fake_swigluOAIAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_swigluoai_and_mul", swigluOAIAndMul, fake_impl=fake_swigluOAIAndMul)


def swigluoai_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT SWIGLUOAI AND MUL is enabled.\033[0m")
    return torch.ops.vllm.nt_swigluoai_and_mul(x)



class NewGELU:
    def arrangement(x,out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, output_arranged
       
       

    def application(x, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        # c = ntl.cast(math.sqrt(2.0 / math.pi), ntl.float32)
        c = ntl.cast(0.7978845608028654, ntl.float32)
        z = c * (x_fp32 + 0.044715 * x_fp32 * x_fp32 * x_fp32)
        tanh_z = libdevice.tanh(z)
        gelu_x = 0.5 * x_fp32 * (1.0 + tanh_z)
        out = gelu_x    

    def premake(ndim: int):
        kernel = ninetoothed.make(
            NewGELU.arrangement,
            NewGELU.application,
            (
                ninetoothed.Tensor(ndim), # x
                ninetoothed.Tensor(ndim), # out
            ),
        )
        return kernel


kernel["newGELU"] = {
        i: NewGELU.premake(i) for i in range(2, 4)
    }

def newGELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out = torch.empty_like(x,dtype=x.dtype, device=x.device)  # 输出形状与输入相同
    kernel["newGELU"][ndim](x, out, BLOCK_SIZE=d)
    return out

def fake_newGELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty_like(x,dtype=x.dtype, device=x.device)  # 输出形状与输入相同
    return out


direct_register_custom_op("nt_gelu_new", newGELU, fake_impl=fake_newGELU)


def gelu_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT NEW GELU is enabled.\033[0m")
    return torch.ops.vllm.nt_gelu_new(x)


class FastGELU:
    def arrangement(x, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, output_arranged
       
       

    def application(x, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        gelu_x = 0.5 * x_fp32 * (1.0 + libdevice.tanh(x_fp32 * 0.7978845608 * (1.0 + 0.044715 * x_fp32 * x_fp32)))
        out = gelu_x

    def premake(ndim: int):
        kernel = ninetoothed.make(
            FastGELU.arrangement,
            FastGELU.application,
            (
                ninetoothed.Tensor(ndim), # x
                ninetoothed.Tensor(ndim), # out
            ),
        )
        return kernel


kernel["fastGELU"] = {
        i: FastGELU.premake(i) for i in range(2, 4)
    }

def fastGELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    kernel["fastGELU"][ndim](x, out, BLOCK_SIZE=d)
    return out


def fake_fastGELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_gelu_fast", fastGELU, fake_impl=fake_fastGELU)



def gelu_fast_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT FAST GELU is enabled.\033[0m")
    return torch.ops.vllm.nt_gelu_fast(x)


class QuickGELU:
    def arrangement(x, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, output_arranged
       
       
    def application(x, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        out = x_fp32 * ntl.sigmoid(1.702 * x_fp32)  
        

    def premake(ndim: int):
        kernel = ninetoothed.make(
            QuickGELU.arrangement,
            QuickGELU.application,
            (
                ninetoothed.Tensor(ndim), # x
                ninetoothed.Tensor(ndim), # out
            ),
        )
        return kernel


kernel["quickGELU"] = {
        i: QuickGELU.premake(i) for i in range(2, 4)
    }

def quickGELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    kernel["quickGELU"][ndim](x, out, BLOCK_SIZE=d)
    return out

def fake_quickGELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_quick_gelu", quickGELU, fake_impl=fake_quickGELU)


def quick_gelu_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT QUICK GELU is enabled.\033[0m")
    return torch.ops.vllm.nt_quick_gelu(x)


class ReLUSquaredActivation:
    def arrangement(x, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, output_arranged
       
       

    def application(x, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        relu_x = ntl.maximum(x_fp32, 0.0)
        squared_x = relu_x * relu_x
        out = squared_x

    def premake(ndim: int):
        kernel = ninetoothed.make(
            ReLUSquaredActivation.arrangement,
            ReLUSquaredActivation.application,
            (
                ninetoothed.Tensor(ndim), # x
                ninetoothed.Tensor(ndim), # out
            ),
        )
        return kernel


kernel["reLUSquaredActivation"] = {
        i: ReLUSquaredActivation.premake(i) for i in range(2, 4)
    }

def reLUSquaredActivation(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    kernel["reLUSquaredActivation"][ndim](x, out, BLOCK_SIZE=d)
    return out


def fake_reLUSquaredActivation(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_relu2", reLUSquaredActivation, fake_impl=fake_reLUSquaredActivation)


def relu2_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT RELUSQUARED ACTIVATION is enabled.\033[0m")
    return torch.ops.vllm.nt_relu2(x)


#XIELU部分

# class TEST:
#       def __init__(self):
#             self.alpha_p = 0.8
#             self.alpha_n = 0.8
#             self.beta = 0.5
#             self.eps = -1e-6
      

class XIELU:
    def arrangement(x, out,alpha_p: float,
                  alpha_n: float,
                  beta: float,
                  eps: float,):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, output_arranged, alpha_p, alpha_n, beta, eps
       
       

    def application(x, out,alpha_p: float,
                  alpha_n: float,
                  beta: float,
                  eps: float,):
        x_f32 = ntl.cast(x, ntl.float32)

        # 正向分支
        pos = alpha_p * x_f32 * x_f32 + beta * x_f32
        
        # 负向分支：clamp x to eps (upper bound for negative side)
        clamped_x = ntl.minimum(x_f32, eps)
        expm1_val = libdevice.expm1(clamped_x)
        neg = (expm1_val - x_f32) * alpha_n + beta * x_f32

        out = ntl.where(x_f32 > 0, pos, neg)
    

    def premake(ndim: int):
        kernel = ninetoothed.make(
            XIELU.arrangement,
            XIELU.application,
            (
                ninetoothed.Tensor(ndim), # x
                ninetoothed.Tensor(ndim), # out
                ninetoothed.Tensor(0), # alpha_p
                ninetoothed.Tensor(0), # alpha_n
                ninetoothed.Tensor(0), # beta
                ninetoothed.Tensor(0), # eps
            ),
        )
        return kernel


kernel["xIELU"] = {
        i: XIELU.premake(i) for i in range(2, 4)
    }

def xIELU(x: torch.Tensor,
                  alpha_p: float,
                  alpha_n: float,
                  beta: float,
                  eps: float, 
                  ) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    kernel["xIELU"][ndim](x, out,alpha_p,alpha_n,beta,eps, BLOCK_SIZE=d)
    return out


def fake_xIELU(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_xielu", xIELU, fake_impl=fake_xIELU)


def xielu_forward(self, x: torch.Tensor) -> torch.Tensor:
    logger.info_once("\033[32mNT XIELU is enabled.\033[0m")
    return torch.ops.vllm.nt_xielu(x)
