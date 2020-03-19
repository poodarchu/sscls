import re

import numpy as np
import torch
from torch._six import container_abcs, int_classes, string_classes

# 'O'       (Python) objects
# 'S', 'a'  zero-terminated bytes (not recommended)
# 'U'       Unicode string
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def numpy_convert(data):
    r"""Converts each NumPy array data field into a tensor"""

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: numpy_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(numpy_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(
        data, string_classes
    ):
        return [numpy_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def numpy_collate(batch, is_merge_batch=True):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    if not is_merge_batch:
        assert len(batch) == 1
        return batch[0]

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype)
                )

            return np.stack([np.array(b) for b in batch], axis=0)
        elif elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(elem, float):
        return np.array(batch, dtype=np.float64)
    elif isinstance(elem, int_classes):
        return np.array(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(numpy_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def pad_collate(batch, is_padding: bool = False, pad_value: float = 0.0):
    elem = batch[0]
    if isinstance(elem, container_abcs.Mapping):
        # dict
        return {
            key: pad_collate(
                [d[key] for d in batch], is_padding=(key == "data")
            )
            for key in elem
        }
    elif isinstance(elem, torch.Tensor):
        if is_padding:
            # img
            max_size = list(max(s) for s in zip(*[img.shape for img in batch]))
            batch_shape = (len(batch),) + tuple(max_size)
            batched_imgs = batch[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(batch, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
            return batched_imgs.contiguous().numpy().astype(np.uint8)
        else:
            # gt_boxes
            max_len = max([len(gt_boxes) for gt_boxes in batch])
            batch_shape = (len(batch),) + (max_len, 5)
            batched_gts = batch[0].new_full(batch_shape, pad_value)
            for gt, pad_gt in zip(batch, batched_gts):
                pad_gt[: gt.shape[-2], :].copy_(gt)
            return batched_gts.contiguous().numpy()
    else:
        return np.stack([np.array(b) for b in batch], axis=0)
