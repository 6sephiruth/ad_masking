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

# normal dataset
_, norm_raw = load_dataset(p.dataset, p.data_dir, normalize=False)
targets = [t for (_,t) in norm_raw]
orig_loader = DataLoader(norm_raw, shuffle=False, **p.kwargs)

# indices for labels
inds = range(0, len(norm_raw), 10)

norm_ds = Subset(norm_raw, inds)
norm_ys = Subset(targets, inds)

norm_loader = DataLoader(norm_ds, shuffle=False, **p.kwargs)

# preprocessing for foolbox
preproc = norm_param(p.dataset, get_axis=True)

# adversarial dataset
adv_xs, adv_idx = attack(model=model,
                            data_loader=norm_loader,
                            method=p.atk_method,
                            epsilon=p.atk_epsilon,
                            preprocessing=preproc,
                            device=device)

# TODO: resolve cuda memory problem
adv_loader = DataLoader(adv_xs, shuffle=False, **p.kwargs)
adv_sub = Subset(adv_xs, adv_idx)
adv_subloader = DataLoader(adv_sub, shuffle=False, **p.kwargs)
tar_sub = Subset(norm_ys, adv_idx)
target_subloader = DataLoader(tar_sub, shuffle=False, **p.kwargs)
target_loader = DataLoader(norm_ys, shuffle=False, **p.kwargs)

norm_sub = Subset(norm_ds, adv_idx)
norm_subloader = DataLoader(norm_sub, shuffle=False, **p.kwargs)

# transform
tf_norm = transforms.Normalize(**norm_param(p.dataset))

# helper function for testing
test_fn = lambda m, d: \
            test(m, d, criterion, 0, device, save_model=False, transform=tf_norm)

# get average attributions
print('calculating average attribution ... ')
attr_norm = get_attr(model=model,
                    data_loader=norm_subloader,
                    attr_method=p.attr_method,
                    transform=tf_norm,
                    device=device)
print('done\n')

# masked model
masked_model = eval(f"Masked{p.model_name}")().to(device)
masked_model.load_state_dict(saved_state['model'])

# different masking portions
N = 20
exclude = ['linear']
print(f"### excluding {exclude} ###")
ks = range(N+1)
for k in ks:
    k = k/N

    # apply mask to the model
    masked_model.masks = get_mask_rand(attr_norm, None, k=k, exclude=exclude)

    print('forwarding on masked model ... ')
    norm_accs = test_fn(masked_model, norm_loader)
    adv_accs = test_fn(masked_model, list(zip(adv_loader, target_loader)))

    # benchmarking
    info = map(str, [p.dataset, p.model_name, p.atk_method, p.atk_epsilon, p.seed])
    stats = map(lambda s: f'{s:.4f}', [k, *norm_accs, *adv_accs])

    # TODO: output best k only

    with open('bench_rand.tsv', 'a') as f:
        f.write('\t'.join([*info, *stats]) + '\n')