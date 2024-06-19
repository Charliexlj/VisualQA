import random
import os
import sys
from tqdm import tqdm

from PIL import Image

import yaml

with open('./src/eval/eval_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

sys.path.append(config["model_path"])

from xraygpt.common.config import Config # type: ignore
from xraygpt.common.registry import registry # type: ignore
from xraygpt.conversation.conversation import Chat, CONV_VISION # type: ignore

# imports modules for registration
from xraygpt.datasets.builders import * # type: ignore# type: ignore
from xraygpt.models import * # type: ignore
from xraygpt.processors import * # type: ignore
from xraygpt.runners import * # type: ignore
from xraygpt.tasks import * # type: ignore


class SimpleArgs:
    def __init__(self, cfg_path, gpu_id=0, options=None):
        self.cfg_path = cfg_path
        self.gpu_id = gpu_id
        self.options = options or []

def parse_args(cfg_path, gpu_id=0):
    return SimpleArgs(cfg_path, gpu_id)

def load_model(args):
    print(args.cfg_path)
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

    return chat

def XrayGPT_Findings(file_paths, xraygpt_cfg, gpu_id=0, direct_path=False):
    args = parse_args(xraygpt_cfg, gpu_id)
    chat = load_model(args)

    findings = []
    
    # for files_folders in file_paths:
    for files_folders in tqdm(file_paths):
        chatbot = []
        chat_state = CONV_VISION.copy()
        img_list = []
        if direct_path:
            file = files_folders
        else:
            files = [file for file in os.listdir(files_folders)]
            file = random.choice(files)
            file = files_folders + '/' + file
        image = Image.open(file).convert('RGB')
        llm_message = chat.upload_img(image, chat_state, img_list)

        user_message = "Describe the given chest x-ray image in detail."
        chat.ask(user_message, chat_state)
        chatbot = chatbot + [[user_message, None]]

        llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=1,
                              temperature=0.1,
                              max_new_tokens=300,
                              max_length=2000)[0]
        # chatbot[-1][1] = llm_message
        findings.append(llm_message)

    return findings
