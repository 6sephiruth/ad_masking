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

# get normal test images
_, norm_raw = load_dataset(p.dataset, p.data_dir, normalize=False)
targets = [t for (_,t) in norm_raw]

# sample indices for labels
L = 10
inds = [[i for i,t in enumerate(targets) if t==l] for l in range(L)]
inds = [np.random.choice(i, int(len(i)/10), replace=False) for i in inds]
inds = sorted(np.concatenate(inds))

# sampled datasets
norm_ds = Subset(norm_raw, inds)
norm_tar = Subset(targets, inds)
norm_loader = DataLoader(norm_ds, shuffle=False, **p.kwargs)

# preprocessing and transformation for foolbox
preproc = norm_param(p.dataset, get_axis=True)
tf_norm = transforms.Normalize(**norm_param(p.dataset))

# helper function for testing
test_fn = lambda m, d: \
            test(m, d, criterion, 0, device, save_model=False, transform=tf_norm)

# adversarial images and indices
adv_ims, adv_idx = attack(model=model,
                          data_loader=norm_loader,
                          method=p.atk_method,
                          epsilon=p.atk_epsilon,
                          preprocessing=preproc,
                          device=device)

# get attributions
print(f'calculating {p.attr_method} ... ')
# subloader for normal dataset
norm_subloader = DataLoader(Subset(norm_ds, adv_idx), shuffle=False,
                            num_workers=4,
                            batch_size=p.attr_bat)
attr_norm = get_attr(model=model,
                    data_loader=norm_subloader,
                    attr_method=p.attr_method,
                    transform=tf_norm,
                    device=device)

# subloaders for adversarial images and targets
adv_subloader = DataLoader(Subset(adv_ims, adv_idx), shuffle=False,
                           num_workers=4,
                           batch_size=p.attr_bat)
target_subloader = DataLoader(Subset(norm_tar, adv_idx), shuffle=False,
                              num_workers=4,
                              batch_size=p.attr_bat)
attr_adv = get_attr(model=model,
                    data_loader=adv_subloader,
                    attr_method=p.attr_method,
                    transform=tf_norm,
                    target_loader=target_subloader,
                    get_y=True,
                    device=device)
print('done\n')

# masked model
masked_model = eval(f"Masked{p.model_name}")().to(device)
masked_model.load_state_dict(saved_state['model'])

# TODO: automatically exclude the last layer
# apply different exclusion per model type
if p.model_name == 'ResNet50':
    #exclude = ['conv1','bn1','layer1','layer2','layer3','linear']
    #exclude = ['linear']
    exclude = ['conv1','bn1','layer1','layer2','linear']
elif p.model_name == 'LeNet':
    #exclude = ['conv2','conv3','flatten','fc1','fc2']
    exclude = ['fc2']

# different masking portions
best_over, best_adv = 0, 0
k_over, k_adv = 0, 0
N = 100
print(f"### excluding {exclude} ###")
ks = range(N+1)
for k in ks:
    k = k/N

    # apply mask to the model
    masked_model.masks = get_mask(attr_norm, attr_adv, k=k, exclude=exclude)

    print('forwarding on masked model ... ')
    norm_acc = test_fn(masked_model, norm_loader)
    adv_loader = DataLoader(adv_ims, shuffle=False, **p.kwargs)
    target_loader = DataLoader(norm_tar, shuffle=False, **p.kwargs)
    adv_acc = test_fn(masked_model, list(zip(adv_loader, target_loader)))

    # record initial stats
    if k == 0:
        stat_init = (norm_acc, adv_acc)

    # record best overall
    if (norm_acc + adv_acc)/2 > best_over:
        best_over = (norm_acc + adv_acc)/2
        stat_over = (k, norm_acc, adv_acc)

    # record best adversarial
    if adv_acc > best_adv:
        best_adv = adv_acc
        stat_adv = (k, norm_acc, adv_acc)

# benchmarking
info = map(str, [p.dataset, p.model_name,
                 p.atk_method, p.atk_epsilon,
                 p.attr_method, p.seed])
stats = map(lambda s: f'{s:.4f}', [*stat_init, *stat_over, *stat_adv])

with open('bench/bench_best.tsv', 'a') as f:
        f.write('\t'.join([*info, *stats]) + '\n')