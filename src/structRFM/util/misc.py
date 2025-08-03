import os
import time
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def str_localtime(str_format="%Y%m%d_%Hh%Mm%Ss"):
    return time.strftime(str_format, time.localtime())


def read_dir(path, suffix_list=None):
    ret = []
    for pre, ds, fs in os.walk(path):
        for f in fs:
            if suffix_list is None or any([f.endswith(suf) for suf in suffix_list]):
                ret.append(os.path.join(pre, f))
    return ret


def get_file_name(path, flag='.', return_suf=False):
    file_name = os.path.basename(path)
    p = file_name.rfind(flag)
    suf = ''
    if p!=-1:
        suf = file_name[p:]
        file_name = file_name[:p]
    if return_suf:
        return file_name, suf
    else:
        return file_name


def timer(func):
    def ret_func(*args, **kargs):
        start = time.time()

        # execute func
        ret = func(*args, **kargs)

        end = time.time()
        res_time = int(end - start)
        units = ['d', 'h', 'm', 's']
        bases = [60*60*24, 60*60, 60, 1]

        nums = [0] * len(units)
        for i in range(len(units)):
            if res_time>bases[i]:
                nums[i] = res_time // bases[i]
                res_time = res_time%bases[i]
        time_str_lst = [f'{num}{unit}' for num, unit in zip(nums, units) if num!=0]
        if len(time_str_lst)>2:
            time_str_lst = time_str_lst[:2]
        time_str = ' '.join(time_str_lst)
        print(f'Time used: {time_str}')
        return ret
    return ret_func


def detect_encoding(path):
    import chardet
    with open(path, 'rb') as f:
        cont = f.read()
        result = chardet.detect(cont)
    return result['encoding']


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
