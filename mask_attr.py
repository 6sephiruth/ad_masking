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


# generate adversarial examples
import foolbox as fb

_, test_raw = load_raw(params_loaded['dataset'], DATA_DIR)      # raw data (0,1)
sub = range(0, len(test_raw), 50)
test_raw_sub = Subset(test_raw, sub)
norm_loader = DataLoader(test_raw_sub, shuffle=False, **kwargs)

preprocessing = load_preproc(params_loaded['dataset'])
fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

def attack(data_loader, fmodel, atk_method='LinfPGD', epsilon=0.1):
    # TODO: make variants
    if atk_method == 'LinfPGD':
        atk = fb.attacks.LinfPGD()

    # output as list
    advs, succ = [], []

    for (x,y) in data_loader:
        _, adv, success = atk(fmodel, x.cuda(), y.cuda(), epsilons=epsilon)
        advs.extend(adv.cpu())
        succ.extend(success.cpu())

    robust_acc = 1 - np.mean(succ)
    print(f"robust accuracy: {robust_acc}")

    return advs

adv_xs = attack(norm_loader, fmodel)
adv_x = DataLoader(adv_xs, shuffle=False, **kwargs)
adv_ys = inference(model, adv_x, device)

# custom dataset
adv_dataset = CustomDataset(adv_xs, adv_ys)
adv_loader = DataLoader(adv_dataset, shuffle=False, **kwargs)

# get average attributions
print('calculating average attribution ... ')
attr_norm = get_attr(norm_loader, model, "LayerActivation")
attr_adv = get_attr(adv_loader, model, "LayerActivation")
print('done\n')


k=0.1
exclude=[]
masks = adv_masks(attr_norm, attr_adv, k=k, exclude=exclude)

print('forwarding on masked model ... ')
total = 0
corr_n, corr_a = 0,0
corr_mask_n, corr_mask_a = 0,0
with torch.no_grad():
    for (x_adv,y_adv),(x,y) in zip(adv_loader,norm_loader):
        # normal forward
        y_p = model(x.cuda())
        _, y_p = torch.max(y_p.data, 1)
        # masked forward
        y_n = masked_forward(model, x, masks)
        y_a = masked_forward(model, x_adv, masks)
        _, y_n = torch.max(y_n.data, 1)
        _, y_a = torch.max(y_a.data, 1)

        # get stats
        corr_n += (y_p.cpu() == y).sum().item()
        corr_a += (y_adv == y).sum().item()
        corr_mask_n += (y_n.cpu() == y).sum().item()
        corr_mask_a += (y_a.cpu() == y).sum().item()

        total += x_adv.size(0)

print('done\n')

# stats
acc_n = corr_n / total
acc_a = corr_a / total
mask_acc_n = corr_mask_n / total
mask_acc_a = corr_mask_a / total

# benchmarking
stats = map(lambda s: f'{s:.4f}', [acc_n, mask_acc_n, acc_a, mask_acc_a])
print(list(stats))
