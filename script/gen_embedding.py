import torch
import torch.nn as nn
import numpy as np
from net.stgcn_embed import Model
from torch.utils.data import DataLoader, TensorDataset


model = Model(in_channels=3, out_channels=400, edge_importance_weighting=True, graph_args={'layout': 'openpose', 'strategy': 'spatial'})
pre_embeddor = torch.load('./pretrain/dqn/embed.pt')
model.load_state_dict(pre_embeddor)
model.to(torch.device("cuda"))
model = nn.DataParallel(model, device_ids=[0,1,2,3])

file_name = "data/Kinetics/train_data.npy"
data = np.load(file_name, mmap_mode='r')
dataset = TensorDataset(torch.tensor(data))
dataloader = DataLoader(dataset, batch_size=128)

model.eval()
start_idx = 0
order = []
output_tensor = torch.empty((len(dataset), 400))
with torch.no_grad():
    for batch in dataloader:
        batch = batch[0]
        batch_size = batch.size(0)
        end_idx = start_idx + batch_size
        embedded = model(batch.to(torch.device("cuda")))
        output_tensor[start_idx:end_idx, :] = embedded
        print(f'{start_idx//128} ----------')
        order.append(start_idx//128)
        start_idx = end_idx

print(all([order[i] == i for i in range(len(order))]))
embedded = np.array(output_tensor)
print(embedded.shape)
np.save('embedded.npy', embedded)
