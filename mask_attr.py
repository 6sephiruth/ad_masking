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

# define subset
sub = range(0, len(test_ds), 10)

# normal dataset
_, norm_raw = load_dataset(p.dataset, p.data_dir, normalize=False)
norm_ds = Subset(norm_raw, sub)
norm_ys = Subset(norm_ds.dataset.targets, sub)
norm_loader = DataLoader(norm_ds, shuffle=False, **p.kwargs)
target_loader = DataLoader(norm_ys, shuffle=False, **p.kwargs)

# preprocessing for foolbox
preproc = norm_param(p.dataset, get_axis=True)

# adversarial dataset
adv_xs = attack(model, norm_loader, p.atk_method, p.atk_epsilon, preproc, device)
adv_ds = TensorDataset(adv_xs, torch.LongTensor(norm_ys))
adv_loader = DataLoader(adv_ds, shuffle=False, **p.kwargs)

for i,x in enumerate(adv_xs):
    plot_img(x, f'a{i}.png')
    if i > 5:
        exit()

# transform
tf_norm = transforms.Normalize(**norm_param(p.dataset))

# helper function for testing
test_fn = lambda m, d: \
            test(m, d, criterion, 0, device, save_model=False, transform=tf_norm)

# get average attributions
print('calculating average attribution ... ')
attr_norm = get_attr(model, norm_loader, p.attr_method, transform=tf_norm)
attr_adv = get_attr(model, adv_loader, p.attr_method, transform=tf_norm)
print('done\n')

# masked model
masked_model = eval(f"Masked{p.model_name}")().to(device)
masked_model.load_state_dict(saved_state['model'])

# different masking portions
N = 10
exclude = ['linear']
ks = range(N+1)
for k in ks:
    k = k/N

    # apply mask to the model
    masked_model.masks = adv_masks(attr_norm, attr_adv, k=k, exclude=exclude, mode='diff')

    print('forwarding on masked model ... ')
    norm_acc = test_fn(masked_model, norm_loader)
    adv_acc = test_fn(masked_model, adv_loader)

    # benchmarking
    info = map(str, [p.dataset, p.model_name, p.atk_method, p.atk_epsilon, p.attr_method])
    stats = map(lambda s: f'{s:.4f}', [k, norm_acc, adv_acc])

    with open('bench.tsv', 'a') as f:
        f.write('\t'.join([*info, *stats]) + '\n')
