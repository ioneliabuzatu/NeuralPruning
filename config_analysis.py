import numpy as np
from copy import deepcopy
import torch
import pandas as pd
from terminaltables import AsciiTable

from utils import *
from models import *
from sensitivity_analysis import SENSITIVITY


class opt():
  model_def = "/content/weight-pruning/config/yolov3-custom.cfg"
  data_config = "/content/weight-pruning/config/custom.data"
  model = '/content/weight-pruning/custom-training-weights/yolov3_ckpt_9_02281713.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_origin = Darknet(opt.model_def).to(device)

data_config = parse_data_config(opt.data_config)
testset_path = data_config["test"]
class_names = load_classes(data_config["names"])

try:
  model_origin.load_state_dict(torch.load(opt.model))
except:
  model_origin.load_darknet_weights(opt.model)

CBL_idx, Conv_idx, prune_idx= prune_utils.parse_module_defs(model_origin.module_defs)
pruning_percentiles_analysis = [5, 10, 20, 30, 40, 50, 60, 70, 80, 88]

final_pruning_percentiles = []

save_layerwise_accuracy = pd.DataFrame(np.nan, index=CBL_idx, columns=pruning_percentiles_analysis)