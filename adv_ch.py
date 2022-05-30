import argparse
import torch.backends.cudnn as cudnn
import numpy as np
import yaml

# custom packages
from utils_ import *
from models import *


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='params_.yaml')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']

# set seed
seed = params_loaded['seed']
np.random.seed(seed)
torch.manual_seed(seed)

# set dataset dir
if 'data_dir' not in params_loaded.keys():
    DATA_DIR = './dataset'
else:
    DATA_DIR = params_loaded['data_dir']

# set model dir
MODEL_DIR = "model/{}/{}_{}.pt".format(params_loaded['dataset'],
                                       params_loaded['dataset'],
                                       params_loaded['model_name'])

# directories for experiments
dirs = [DATA_DIR, f"./model/{params_loaded['dataset']}"]
makedirs(dirs)

# set cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# set train/test keyword arguments
kwargs = {'batch_size': params_loaded['batch_size']}
if use_cuda:
    cuda_kwargs = {'num_workers': 2,
                   'pin_memory': True}
    kwargs.update(cuda_kwargs)

# load datasets
train_loader, test_loader = load_data(params_loaded['dataset'], DATA_DIR, kwargs)

# setup model
model = eval(params_loaded['model_name'])().to(device)

# load criterion, optimizer, scheduler
criterion, optimizer, scheduler = load_setup(model.parameters(),
                                             params_loaded['dataset'],
                                             params_loaded['learning_rate'])

# load pretrained model
try:
    saved_state = torch.load(MODEL_DIR)
    model.load_state_dict(saved_state['model'])
    print('[*] best_acc:', saved_state['acc'])
    print('[*] best_epoch:', saved_state['epoch'])
    model.eval()
except:
    print('error!')
    exit()

### import adversarial packages ###
from adv_util import *

atk_method = 'PGD'

sub = range(0, len(e.test), 20)
adv_test = Subset(adv_ds, sub)

adv_xs, adv_ys = attack(test_loader, model, atk_method, 0.03)
