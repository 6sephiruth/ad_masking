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

### adversarial package: foolbox ###
import foolbox as fb

### preprocessed data, epsilon ###
mx = (1 - 0.1307)/0.3081 + 1e-6
mi = (0 - 0.1307)/0.3081 - 1e-6

fmodel = fb.PyTorchModel(model, bounds=(mi,mx))

atk = fb.attacks.LinfPGD()

epsilons = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]
epsilons = np.array(epsilons)/0.3081
for (x,y) in test_loader:
    _, advs, success = atk(fmodel, x.cuda(), y.cuda(), epsilons=epsilons)

    robust_accuracy = 1 - success.type(torch.float32).mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    for i in range(5):
        plot_img(x[i], f'o{i}.png')
        plot_img(advs[3][i].detach().cpu(), f'adv0{i}.png')
        print(success[3][i])

    break


### raw data, epsilon ###
_, test_raw = load_raw(params_loaded['dataset'], DATA_DIR)      # raw data (0,1)
test_loader = DataLoader(test_raw, shuffle=False, **kwargs)

preprocessing = load_preproc(params_loaded['dataset'])
fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

atk = fb.attacks.LinfPGD()
epsilons = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]
for (x,y) in test_loader:
    _, advs, success = atk(fmodel, x.cuda(), y.cuda(), epsilons=epsilons)

    robust_accuracy = 1 - success.type(torch.float32).mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    for i in range(5):
        plot_img(x[i], f'o{i}.png')
        plot_img(advs[3][i].detach().cpu(), f'adv1{i}.png')
        print(success[3][i])

    break