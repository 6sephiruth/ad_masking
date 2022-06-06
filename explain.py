# custom packages
from utils import *
from models import *

# get parameters from input file
p = get_params()

# shortcut for device
device = p.device

# load datasets
train_ds, test_ds = load_dataset(p.dataset, p.data_dir)

# make dataloaders
train_loader = DataLoader(train_ds, shuffle=True, **p.kwargs)
test_loader = DataLoader(test_ds, shuffle=False, **p.kwargs)

# setup model
model = eval(p.model_name)().to(device)

# load criterion, optimizer, scheduler
criterion, optimizer, scheduler = load_setup(model.parameters(),
                                             p.dataset,
                                             p.learning_rate)

try:
    # load pretrained model
    saved_state = torch.load(p.model_dir)
    model.load_state_dict(saved_state['model'])
    print('[*] best_acc:', saved_state['acc'])
    print('[*] best_epoch:', saved_state['epoch'])
    model.eval()
except:
    print('error!')
    exit()

### captum packages ###

from captum.attr import (
    LayerActivation,
    LayerConductance,
    InternalInfluence,
    LayerGradientXActivation,
    LayerGradCam,
    LayerDeepLift,
    LayerDeepLiftShap,
    LayerGradientShap,
    LayerIntegratedGradients,
    LayerFeatureAblation,
    LayerLRP
)

attr_method = "LayerActivation"

# aggregate attribution for each layer
avgs = {}
for n,l in model.named_children():
    res = []
    attr = eval(attr_method)(model, l)
    for (x,y) in test_loader:
        if attr_method == 'LayerActivation':
            res.append(attr.attribute(x.cuda()).detach().cpu())
        else:
            res.append(attr.attribute(x.cuda(), target=y.cuda()).detach().cpu())

    res = torch.cat(res, 0)
    avgs[n] = torch.mean(res, 0, True)
    print(n, avgs[n].size(), avgs[n])
    exit()

print(avgs)