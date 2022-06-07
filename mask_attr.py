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

test_loader_ = test_loader

# set test dataset
_, test_raw = load_dataset(p.dataset, p.data_dir, normalize=False)      # raw data (0,1)
sub = range(0, len(test_raw), 1)
test_raw = Subset(test_raw, sub)
norm_loader = DataLoader(test_raw, shuffle=False, **p.kwargs)
preproc = norm_param(p.dataset, get_axis=True)

adv_xs = attack(norm_loader, model, p.atk_method, p.atk_epsilon, preproc, device)
adv_x = DataLoader(adv_xs, shuffle=False, **p.kwargs)
adv_ys = inference(model, adv_x, device)
norm_ys = test_raw.dataset.targets
norm_ys = Subset(norm_ys, sub)

# custom dataset
adv_dataset = CustomDataset(adv_xs, norm_ys)
adv_loader = DataLoader(adv_dataset, shuffle=False, **p.kwargs)

# get average attributions
print('calculating average attribution ... ')
attr_norm = get_attr(norm_loader, model, p.attr_method)
attr_adv = get_attr(adv_loader, model, p.attr_method)
print('done\n')

# masked model
masked_model = eval(f"Masked{p.model_name}")().to(device)
masked_model.load_state_dict(saved_state['model'])

# different masking portions
N = 10
ks = range(N+1)
for k in ks:
    k = k/N

    # apply mask to the model
    masked_model.masks = adv_masks(attr_norm, attr_adv, k=k, exclude=[])

    print('forwarding on masked model ... ')
    norm_acc = test(masked_model, norm_loader, criterion, 0, device, save_model=False)
    adv_acc = test(masked_model, adv_loader, criterion, 0, device, save_model=False)

    # benchmarking
    info = map(str, [p.dataset, p.model_name, p.atk_method, p.atk_epsilon, p.attr_method])
    stats = map(lambda s: f'{s:.4f}', [k, norm_acc, adv_acc])

    with open('bench.tsv', 'a') as f:
        f.write('\t'.join([*info, *stats]) + '\n')

"""
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
"""