import requests
from torch import Tensor, device
from typing import Tuple, List
from tqdm import tqdm
import sys

def batch_to_device(batch: Tuple[List[Tensor], List[Tensor], List[Tensor], Tensor], target_device: device) \
        -> Tuple[List[Tensor], List[Tensor], List[Tensor], Tensor]:
    """
    send a batch to a device

    :param batch:
    :param target_device:
    :return: the batch sent to the device
    """
    inputs = [t.to(target_device) for t in batch[0]]
    segments = [t.to(target_device) for t in batch[1]]
    masks = [t.to(target_device) for t in batch[2]]
    labels = batch[3].to(target_device)
    return inputs, segments, masks, labels

def http_get(url, path):
    file_binary = open(path, "wb")
    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()

    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total, unit_scale=True)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            file_binary.write(chunk)
    progress.close()