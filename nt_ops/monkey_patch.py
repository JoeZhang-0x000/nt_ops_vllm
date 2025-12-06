import importlib
from vllm.logger import init_logger
import nt_ops

logger = init_logger(__name__)

# A list of tuples defining the patches to apply.
# Each tuple contains:
# (module_path, object_name, attribute_to_patch, patch_function)
# If object_name is None, the attribute is patched directly on the module.
_PATCHES = [
    ('vllm.model_executor.layers.utils', None, 'dispatch_unquantized_gemm',
     nt_ops.linear.dispatch_unquantized_gemm),
    ('vllm.model_executor.layers.activation', 'SiluAndMul', 'forward',
     nt_ops.activation.silu_and_mul_forward),
     ('vllm.model_executor.layers.activation', 'FatreluAndMul', 'forward',
     nt_ops.activation.fatrelu_and_mul_forward),
    ('vllm.model_executor.layers.activation', 'MulAndSilu', 'forward',
     nt_ops.activation.mul_and_silu_forward),
    ('vllm.model_executor.layers.activation', 'GeluAndMul', 'forward',
     nt_ops.activation.gelu_and_mul_forward),
    ('vllm.model_executor.layers.activation', 'SwigluOAIAndMul', 'forward',
     nt_ops.activation.swigluoai_and_mul_forward),
    ('vllm.model_executor.layers.activation', 'NewGELU', 'forward',
     nt_ops.activation.gelu_new_forward),
    ('vllm.model_executor.layers.activation', 'FastGELU', 'forward',
     nt_ops.activation.gelu_fast_forward),
     ('vllm.model_executor.layers.activation', 'QuickGELU', 'forward',
     nt_ops.activation.quick_gelu_forward),
     ('vllm.model_executor.layers.activation', 'ReLUSquaredActivation', 'forward',
     nt_ops.activation.relu2_forward),
     ('vllm.model_executor.layers.activation', 'XIELU', 'forward',
     nt_ops.activation.xielu_forward),
    ('vllm.model_executor.layers.layernorm', 'RMSNorm', 'forward',
     nt_ops.rms.rms_forward),
     ("vllm.attention.ops.triton_unified_attention", None, "unified_attention",
     nt_ops.attention.unified_attention_2d),
]

_patches_applied = False


def apply_monkey_patches():
    """
    Applies all monkey patches defined in the _PATCHES list.
    This function is designed to be executed only once.
    """
    global _patches_applied
    if _patches_applied:
        logger.warning("\033[33mWarning: Monkey patches have already been applied. Skipping.\033[0m")
        return

    for module_path, obj_name, attr_name, patch_func in _PATCHES:
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the target object to patch
            target_obj = getattr(module, obj_name) if obj_name else module
            target_name = f"{module_path}.{obj_name}" if obj_name else module_path

            # Check if the specific attribute is already patched
            if getattr(target_obj, attr_name) is patch_func:
                logger.warning(
                    f"\033[33mWarning: {target_name}.{attr_name} is already patched. Skipping.\033[0m"
                )
                continue

            # Apply the patch
            setattr(target_obj, attr_name, patch_func)

            logger.info(f"\033[31mSuccessfully patched {target_name}.{attr_name}.\033[0m")

        except (ImportError, AttributeError) as e:
            logger.error(f"\033[31mFailed to apply patch for {module_path}.{obj_name}.{attr_name}: {e}\033[0m")

    _patches_applied = True


# Apply all patches when this module is imported.
apply_monkey_patches()