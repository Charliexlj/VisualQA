import yaml
import sys

with open('./src/eval/eval_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
sys.path.append(config["package_path1"])
sys.path.append(config["package_path2"])
sys.path.append(config["package_path3"])
sys.path.append(config["package_path4"])


from .radiology import MIMIC_Findings
from .chexbert import label as CheXbert_label
from .xraygpt import XrayGPT_Findings
from .llava_med import LlavaMed_Findings
# from .radfm import RadFM_Findings
from .vlm import VLM_Findings
