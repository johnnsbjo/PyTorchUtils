# %%
from typing import OrderedDict 
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.serialization import load
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

#%%

def oldloop():
  for i in range(1, epochs+1):
    for train, val in zip(train_loader, val_loader):
      train_labels = torch.stack((((train[1] - 1)**2), train[1]), dim=1).to(dtype=torch.float, device='cuda')
      val_labels = torch.stack((((val[1] - 1)**2), val[1]), dim=1).to(dtype=torch.float, device='cuda')

      train_output = model.model(train[0].view(train[0].shape[0], -1))
      train_loss = loss_fn(train_output, train_labels)

      val_output = model.model(val[0].view(val[0].shape[0], -1))
      val_loss = loss_fn(val_output, val_labels)

      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
    if i % round(epochs/10) == 0:
      print(f"Epoch: {i}, Train Loss: {train_loss}, Val Loss: {val_loss}")

class Net(nn.Module):
  def __init__(self, n_chanels):
    super().__init__()
    self.conv1 = nn.Conv2d(3, n_chanels, kernel_size=3, padding=1)
    self.act1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2)
    self.conv1_dropout = nn.Dropout2d(p=0.2)
    self.conv2 = nn.Conv2d(n_chanels, n_chanels * 2, kernel_size=3, padding=1)
    self.act2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2)
    self.conv2_dropout = nn.Dropout2d(p=0.1)
    self.conv3 = nn.Conv2d(n_chanels * 2, 8, kernel_size=3, padding=1)
    self.act3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(8 * 4 * 4, 64)
    self.act4 = nn.ReLU()
    self.fc2 = nn.Linear(64, 2)

  def to(self, *args, **kwargs):
    return super().to(*args, **kwargs)

  def forward(self, x):
    out = self.pool1(self.act1(self.conv1(x)))
    out = self.conv1_dropout(out)
    out = self.pool2(self.act2(self.conv2(out)))
    out = self.conv2_dropout(out)
    out = self.pool3(self.act3(self.conv3(out)))
    out = out.view(-1, 8 * 4 * 4) 
    out = self.act4(self.fc1(out))
    out = self.fc2(out)
    return out


#%%
class CustomModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(3072, 3072)
    self.tanh1 = nn.Tanh()
    self.linear2 = nn.Linear(3072, 2)
    self.tanh2 = nn.Tanh()

  def to(self, *args, **kwargs):
    return super().to(*args, **kwargs)

  def forward(self, x):
    out = self.linear1(x)
    out = self.tanh1(out)
    out = self.linear2(out)
    out = self.tanh2(out)
    return out


def training_loop(n_epochs: int, model, train_loader: DataLoader, 
  optimizer, loss_fn, device, val_loader=False, verbose=False, **kwargs) -> OrderedDict:
  best = OrderedDict()
  last_val_loss = 10e20
  for i in range(1, n_epochs+1):
    loader_iter = zip(train_loader, val_loader) if val_loader else train_loader
    for x in loader_iter:

      train_inputs, train_labels = x[0]
      test_inputs, test_labels = x[1] if len(x) > 1 else (None, None)

      #pytorch is converting my list of tuples nx2 into a len 2 list of n size tensors
      #train_labels = torch.stack(train_labels, dim=1).to(device=device)
      #test_labels = torch.stack(test_labels, dim=1).to(device=device) if test_labels else None
      model = model.train()
      train_output = model(train_inputs)
      train_loss = loss_fn(train_output, train_labels)

      if test_inputs is not None:
        with torch.no_grad():
          model=model.eval()
          val_output = model(test_inputs)
          val_loss = loss_fn(val_output, test_labels)

      model = model.train()
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
    
    if val_loss < last_val_loss:
      best = model.state_dict()
      last_val_loss = val_loss

    if verbose and (i % round(n_epochs/10) == 0):
      print(f"Epoch: {i}, Train Loss: {train_loss}, Val Loss: {val_loss}")
  return best


class CustomDataset(Dataset):
    def __init__(self, labels, data, transform=None, target_transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.labels[idx]
        if self.transform:
            d = self.transform(d)
        if self.target_transform:
            l = self.target_transform(l)
        return d, l
#%%
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
               
data_path = '../data-unversioned/p1ch7/'
_device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

#%%
means = torch.tensor((0.4915, 0.4823, 0.4468)).to(device=_device)

st_devs = torch.tensor((0.2470, 0.2435, 0.2616)).to(device=_device)

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                          transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(means, st_devs)]))

#%%
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']

tensor_cifar2 = [(img, label_map[label]) for img, label in tensor_cifar10 if label in [0,2]]

n_samples = tensor_cifar2.__len__()
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

tensor_cifar2_train = [tensor_cifar2[i] for i in train_indices]
tensor_cifar2_val = [tensor_cifar2[i] for i in val_indices]

train_images, train_labels = zip(*[(img, label) for img, label in tensor_cifar2_train])
val_images, val_labels = zip(*[(img, label) for img, label in tensor_cifar2_val])

train_images = torch.stack(train_images, dim=0).to(device=_device, dtype=torch.float)
val_images = torch.stack(val_images, dim=0).to(device=_device, dtype=torch.float)

train_labels = torch.tensor(train_labels).to(device=_device, dtype=torch.long)
val_labels = torch.tensor(val_labels).to(device=_device, dtype=torch.long)

tensor_cifar2_ds_train = CustomDataset(labels=train_labels, data=train_images)

tensor_cifar2_ds_val = CustomDataset(labels=val_labels, data=val_images)

#%%
batch_size = 64

train_loader : DataLoader = DataLoader(tensor_cifar2_ds_train, 
                                          batch_size=batch_size,
                                          shuffle=True)
                                           
val_loader : DataLoader = DataLoader(tensor_cifar2_ds_val, 
                                          batch_size=batch_size,
                                          shuffle=False)                         
# %%
model = Net(n_chanels=64)
model = model.to(device=_device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# %%
epochs = 500

loss_fn = nn.CrossEntropyLoss()

result = training_loop(
  n_epochs=epochs,
  train_loader=train_loader,
  val_loader=val_loader,
  model=model,
  loss_fn=loss_fn,
  optimizer=optimizer,
  device = _device,
  verbose=True
)


#%%

# %%
correct = 0
total = 0

with torch.no_grad():
  for imgs, labels in val_loader:
    labels = labels.to(device='cuda')
    batch_size = imgs.shape[0]
    val_model = Net(n_chanels=64)
    val_model.load_state_dict(result)
    val_model = val_model.to(device=_device)
    outputs = val_model(imgs)
    _, predicted = torch.max(outputs,dim=1)
    total += labels.shape[0]
    correct += int((predicted == labels).sum())

print(correct, total, correct/total)
torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt')