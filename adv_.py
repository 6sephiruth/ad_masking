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


### raw data, epsilon ###
_, test_raw = load_dataset(p.dataset, p.data_dir, normalize=False)      # raw data (0,1)
sub = range(0, len(test_raw), 50)
test_raw_sub = Subset(test_raw, sub)
test_loader = DataLoader(test_raw_sub, shuffle=False, **p.kwargs)
preproc = norm_param(p.dataset, get_axis=True)

adv_xs = attack(test_loader, model, p.atk_method, p.atk_epsilon, preproc, device)
print(adv_xs)
print(adv_xs.size())

"""
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
"""