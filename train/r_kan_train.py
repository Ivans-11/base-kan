import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('..'))
from kans import RationalKAN
from kans.utils import create_dataset

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# �������ݼ�
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
print('train_input size:', dataset['train_input'].shape)
print('train_label',dataset['train_label'].shape)
print('test_input size:', dataset['test_input'].shape)
print('test_label',dataset['test_label'].shape)

# �������ݼ�����
train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ����ģ��
layer_sizes = [2,5,3,1]  # ָ��ÿ��Ľڵ���
mole_order = 5  # ���ӽ���
deno_order = 4  # ��ĸ����

model = RationalKAN(layer_sizes, mole_order, deno_order)
model.to(device)

# �Ż�������ʧ����
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# ѵ������
num_epochs = 50
epoch_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # ǰ�򴫲�
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # ���򴫲����Ż�
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ��¼��ʧ
        running_loss += loss.item()
        
        # ÿ��һ��������ӡһ����Ϣ
        # if (i + 1) % 10 == 0:
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # ÿ��epoch���������ƽ����ʧ
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}")
    epoch_losses.append(epoch_loss)

# ����ģ��
torch.save(model.state_dict(), 'model/rational_kan_model.pth')

# ������ʧ����
plt.figure(figsize=(8,6))
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss of r_kan model')
plt.grid(True)
plt.show()

# ����ģ��
# model = RationalKAN(layer_sizes, mole_order, deno_order)
# model.load_state_dict(torch.load('model/rational_kan_model.pth'))

# ����ģ��
model.eval()
test_input = dataset['test_input']
test_label = dataset['test_label']
with torch.no_grad():
	test_output = model(test_input)
test_loss = criterion(test_output, test_label).item()
print(f"Test Loss: {test_loss:.4f}")