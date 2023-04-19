import torch
from stimulus import StimGenerator
from model import BI_RNN
from tqdm import trange
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

epoch = 64
batch = 10
batch_sz = 64
model = BI_RNN(n_cell=8)
sg = StimGenerator(dt = .1)
optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3
        )
# Training loop

loss = []
for ep in trange(epoch):
    for ib in range(batch):
        inp, out = sg.get_batch_data(batch_sz)
        inp, out = sg.extend_sim(30, inp, out)
        # inp += torch.randn_like(inp)*.01
        # out += torch.randn_like(out)*.01
        # out[...] = 1

        optimizer.zero_grad()
        l = model.loss(inp, out)
        with torch.autograd.set_detect_anomaly(True):
            l.backward()
            optimizer.step()
        print(l)
        loss.append(l.detach())
        with torch.no_grad():
            inp, out = sg.get_batch_data(2)
            inp, out = sg.extend_sim(30, inp, out)
            # inp += torch.randn_like(inp)*.01
            # out += torch.randn_like(out)*.01
            # out *= 0
            pred = model(inp)
            plt.plot(inp[0,:])
            plt.plot(out[0,:])
            plt.plot(pred[0,:])
            plt.show(block = False)
            plt.pause(.01)
            plt.clf()

plt.plot(loss)
plt.show()