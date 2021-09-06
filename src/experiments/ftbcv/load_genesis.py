
"""
https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch
"""

"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
train_loader = DataLoader(Your Dataset, batch_size=config.batch_size, shuffle=True)

# prepare the 3D model

model = unet3d.UNet3D()

#Load pre-trained weights
weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(intial_epoch, config.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
