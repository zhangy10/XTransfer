import torch
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Callable, Dict, Optional, List

from xtransfer.tools import mmc, build_model_dict
from utils import save_dict, load_dict

activation_ = defaultdict()
out_ = defaultdict(list)


def get_input_hook(name):
    in_name = '{}_in'.format(name)

    def hook(model, input, output):
        global activation_
        activation_[in_name] = input[0].detach().cpu().numpy()

    return hook


def get_output_hook(name):
    out_name = '{}_out'.format(name)

    def hook(model, input, output):
        global activation_
        activation_[out_name] = output.detach().cpu().numpy()

    return hook


def get_mask_hook(conv):
    def hook(model, input, output):
        masked_output = output * conv.mask
        return masked_output

    return hook


def get_activation_hook(name, dict):
    def hook(model, input, output):
        dict[name] = output.detach().cpu()

    return hook


def save_named_activation(mode='mean-split', num_split=10):
    global activation_
    global out_
    for layer_id, item in activation_.items():
        if mode == 'mean-split':
            out_[layer_id].append(np.mean(np.split(item, num_split), axis=1))
        elif mode == 'og':
            out_[layer_id].append(item[:])
        elif mode == 'mmc':
            out_[layer_id].append(mmc(item))
        else:
            raise NotImplementedError
    clear()
    return out_


def clear():
    global activation_
    activation_ = defaultdict()


def clear_out():
    global out_
    out_ = defaultdict(list)


def register_named_fw_hook(model, input_names=None, output_names=None, **kwargs):
    """
    register hook method
    """
    model_dict = build_model_dict(model)
    hook_handles = []
    if input_names:
        for name in input_names:
            layer = model_dict[name]
            handle = layer.register_forward_hook(get_input_hook(name))
            hook_handles.append(handle)
    if output_names:
        for name in output_names:
            layer = model_dict[name]
            handle = layer.register_forward_hook(get_output_hook(name))
            hook_handles.append(handle)
    return hook_handles


def get_dict():
    global out_
    return out_


def add_to_dict(key, item):
    global out_
    out_[key] = item


def add_label(label):
    global out_
    out_['label'] = label


def register_obj_hook(model, obj, outD=True, inD=False):
    hook_handles = []
    for name, layer in model.named_modules():
        if isinstance(layer, obj):
            if outD:
                handle = layer.register_forward_hook(get_output_hook(name))
                hook_handles.append(handle)
            if inD:
                handle = layer.register_forward_hook(get_input_hook(name))
                hook_handles.append(handle)

    return hook_handles


def dump_dic(path, mode='mean-split'):
    global out_
    if mode == 'mean-split':
        for key, item in out_.items():
            dd = np.stack(item).squeeze()
            dd = np.reshape(dd, list([-1] + list(dd.shape[2:])))
            out_[key] = dd
    elif mode == 'og':
        for key, item in out_.items():
            if hasattr(item[0], '__iter__'):
                dd = np.concatenate(item, axis=0)
            else:
                dd = item
            out_[key] = dd
    elif mode == 'single':
        for key, item in out_.items():
            dd = item
            out_[key] = dd
    save_dict(out_, path)
    print('activation data has been saved to {}'.format(path))
    out_ = defaultdict(list)


def _remove_all_forward_hooks(
        module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.

    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safely identify and remove
    the target forward hooks.
    ref: https://gist.github.com/ProGamerGov/e4060b55c702835ac933d95f063a2f6e

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        hook_fn_name (str, optional): Optionally only remove specific forward hooks
            based on their function's __name__ attribute.
            Default: None
    """

    # if hook_fn_name is None:
    #     warn("Removing all active hooks will break some PyTorch modules & systems.")

    def _remove_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(module, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                if name is not None:
                    dict_items = list(m._forward_hooks.items())
                    m._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != name]
                    )
                else:
                    m._forward_hooks: Dict[int, Callable] = OrderedDict()

    def _remove_child_hooks(
            target_module: torch.nn.Module, hook_name: Optional[str] = None
    ) -> None:
        for name, child in target_module._modules.items():
            if child is not None:
                _remove_hooks(child, hook_name)
                _remove_child_hooks(child, hook_name)

    # Remove hooks from target submodules
    _remove_child_hooks(module, hook_fn_name)

    # Remove hooks from the target module
    _remove_hooks(module, hook_fn_name)


def _count_forward_hooks(
        module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> int:
    """
    Count the number of active forward hooks on the specified module instance.
    ref: https://gist.github.com/ProGamerGov/e4060b55c702835ac933d95f063a2f6e
    Args:

        module (nn.Module): The model module instance to count the number of
            forward hooks on.
        name (str, optional): Optionally only count specific forward hooks based on
            their function's __name__ attribute.
            Default: None

    Returns:
        num_hooks (int): The number of active hooks in the specified module.
    """

    num_hooks: List[int] = [0]

    def _count_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(m, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                dict_items = list(m._forward_hooks.items())
                for i, fn in dict_items:
                    if hook_fn_name is None or fn.__name__ == name:
                        num_hooks[0] += 1

    def _count_child_hooks(
            target_module: torch.nn.Module,
            hook_name: Optional[str] = None,
    ) -> None:

        for name, child in target_module._modules.items():
            if child is not None:
                _count_hooks(child, hook_name)
                _count_child_hooks(child, hook_name)

    _count_child_hooks(module, hook_fn_name)
    _count_hooks(module, hook_fn_name)
    return num_hooks[0]

