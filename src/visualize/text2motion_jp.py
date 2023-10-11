import os
import sys

sys.path.append('.')
import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.31.54', port=17777, stdoutToServer=True, stderrToServer=True)
import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_clip_text, get_gpu_device
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa
import codecs as cs
from os.path import join as pjoin
from tqdm import tqdm

plt.switch_backend('agg')


def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    model, datasets = get_model_and_data(parameters, split='vald')

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    texts = []
    keyids = os.listdir('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_1007/prompt_0')
    keyids = [item.split('.')[0] for item in keyids if item.endswith('.json')]
    for keyid in keyids:
        with cs.open(pjoin('/mnt/disk_1/jinpeng/T2M/data/HumanML3D/texts', keyid + '.txt')) as f:
            line_split = f.readline().split('#')
            texts.append(line_split[0])
    grid = []
    for i in range(len(texts) // 4 + (len(texts) % 4 != 0)):
        grid.append(texts[i * 4:(i + 1) * 4])
    grid[-1] += [''] * (4 - len(grid[-1]))
    grid_keyid = []
    for i in range(len(keyids) // 4 + (len(keyids) % 4 != 0)):
        grid_keyid.append(keyids[i * 4:(i + 1) * 4])
    grid_keyid[-1] += [''] * (4 - len(grid_keyid[-1]))
    for grid_item, grid_keyid_item in tqdm(zip(grid, grid_keyid)):
        viz_clip_text(model, grid_item, grid_keyid_item, epoch, parameters, folder='/mnt/disk_1/jinpeng/T2M/results'
                                                                                   '/motionclip')


if __name__ == '__main__':
    main()
