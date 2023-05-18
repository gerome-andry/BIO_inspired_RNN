import torch
from stimulus import StimGenerator
from BIRNN import ResMLP, SenseMemAct, decode_choice, encode_choice
from tqdm import trange
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

epoch = 64
batch = 64
batch_sz = 128
memory_size = 128
in_emb = memory_size//4
mem_lay = 1
inputs_dim = 2
decisions = 3
CELL = 'GRU'

sensor = ResMLP(inputs_dim, in_emb, [64,64,64])
actor = ResMLP(memory_size, decisions, [64,64,64])

model = SenseMemAct(sensor, actor, in_sz=in_emb, mem_sz=memory_size, 
                    mem_lay=mem_lay, decisions=decisions, type = CELL).cuda()

sg = StimGenerator(dt = .1)
optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3
        )
# Training loop

choice_d = {
    0 : 'none',
    1 : 'same',
    2 : 'diff'
}

loss = []
for ep in trange(epoch):
    for ib in range(batch):
        inp, out = sg.get_batch_data(batch_sz)
        inp, out = sg.extend_sim(30, inp, out)
        # inp += torch.randn_like(inp)*.01
        # out += torch.randn_like(out)*.01
        # out[...] = 1

        optimizer.zero_grad()
        B,L = inp.shape
        mod_in = torch.zeros((B,L,2))
        mod_in[...,0][inp == 1] = 1
        mod_in[...,1][inp != 1] = inp[inp!=1]

        l = model.loss(mod_in.cuda(), decode_choice(out).cuda())
        with torch.autograd.set_detect_anomaly(True):
            l.backward()
            optimizer.step()
        print(l)
        loss.append((l.detach()).cpu())
        with torch.no_grad():
            inp, out = sg.get_batch_data(2)
            inp, out = sg.extend_sim(30, inp, out)
            # inp += torch.randn_like(inp)*.01
            # out += torch.randn_like(out)*.01
            # out *= 0
            B,L = inp.shape
            mod_in = torch.zeros((B,L,2))
            mod_in[...,0][inp == 1] = 1
            mod_in[...,1][inp != 1] = inp[inp!=1]

            pred = model(mod_in.cuda()).cpu()

            pred = encode_choice(pred)
            plt.plot(inp[0,:])
            plt.title('pred')
            plt.plot(out[0,:])
            plt.plot(pred[0,:])
            plt.show(block = False)
            plt.pause(.01)
            plt.clf()

plt.plot(loss)
plt.show()

torch.save(model.state_dict(), './checkpoint_GRU.pth')
