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
    # try to load pretrained model
    saved_state = torch.load(p.model_dir)
    model.load_state_dict(saved_state['model'])
    print('[*] best_acc:', saved_state['acc'])
    print('[*] best_epoch:', saved_state['epoch'])

except:
    # train model
    best_acc = 0
    for epoch in range(1, p.n_epoch + 1):
        train(model, train_loader, optimizer, criterion, epoch, device)
        best_acc = test(model, test_loader, criterion, epoch, device, best_acc)
        scheduler.step()

    # save model with the best accuracy
    torch.save(torch.load('./checkpoint/ckpt.pth'), p.model_dir)