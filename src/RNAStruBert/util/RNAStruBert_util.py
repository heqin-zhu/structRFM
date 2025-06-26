import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def rename_para(checkpoint_dir:str):
    '''
        checkpoint_dir contains 'model.safetensors'
    '''
    checkpoint_path = os.path.join(checkpoint_dir, 'model.safetensors')
    tensors = {}
    metadata_dic = None
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata_dic = f.metadata()  # save metadata
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    new_tensors = {}
    for key, tensor in tensors.items():
        if key.startswith("bert.encoder."):
            new_key = key.replace("bert.encoder.", "encoder.", 1)  # only replace the first one
            new_tensors[new_key] = tensor
        else:
            new_tensors[key] = tensor
    dest_dir = os.path.join(os.path.dirname(checkpoint_dir), os.path.basename(checkpoint_dir)+'_rename_para')
    dest_path = os.path.join(dest_dir, 'model.safetensors')
    os.makedirs(dest_dir, exist_ok=True)
    for f in os.listdir(checkpoint_dir):
        if not os.path.exists(os.path.join(dest_dir, f)):
            shutil.copy(os.path.join(checkpoint_dir, f), dest_dir)
    save_file(new_tensors, dest_path, metadata=metadata_dic)
