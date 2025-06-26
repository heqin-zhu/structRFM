import os
import shutil

from safetensors.torch import load_file, save_file


def rename_para_name(checkpoint_dir:str):
    '''
        checkpoint_dir contains 'model.safetensors'
    '''
    checkpoint_path = os.path.join(checkpoint_dir, 'model.safetensors')
    state_dict = load_file(checkpoint_path)

    dest_dir = os.path.join(os.path.dirname(checkpoint_dir), os.path.basename(checkpoint_dir)+'_rename_para')
    os.makedirs(dest_dir, exist_ok=True)
    for f in os.listdir(checkpoint_dir):
        shutil.copy(os.path.join(checkpoint_dir, f), dest_dir)

    new_state_dict = {}
    rename_count = 0

    for key, value in state_dict.items():
        if key.startswith("bert.encoder."):
            new_key = "encdoer." + key[len("bert.encoder."):]
            rename_count += 1
        else:
            new_key = key
        new_state_dict[new_key] = value
    save_file(new_state_dict, os.path.join(dest_dir, 'model.safetensors'))
