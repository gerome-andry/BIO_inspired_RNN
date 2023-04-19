import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_lay, hidden):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim,hidden)])
        self.layers.extend([nn.Linear(hidden, hidden) for _ in range(n_lay)])
        self.layers.append(nn.Linear(hidden, out_dim))
        self.act = nn.ReLU()

    def forward(self, x, out_act = True):
        for l in self.layers:
            x = self.act(l(x))
        
        if out_act:
            x = self.act(x)

        return x

class BI_cell(nn.Module):
    def __init__(self, hidden = 64, lay = 4, num_mem = 1):
        super().__init__()
        # parameters must always be > 0
        self.sense = MLP(1 + 2*num_mem + 0*num_mem, (1 + 5)*num_mem, lay, hidden)
        self.act = nn.ReLU()
        self.nl = nn.Tanh()
        self.ensure_gate = nn.Sigmoid()
        self.n_mem = num_mem
        self.register_buffer('params', None)

    def forward(self, h, u):
        # print(h,u)
        if self.params is None:
            self.params = torch.randn((h.shape[0], self.n_mem*5))
        # print(self.params.shape)
        pars = torch.cat((u, h), dim = 1)

        u_emb,a, \
        b,c, \
        d,e = torch.split(self.sense(pars), self.n_mem, dim = 1)
        # print(a.shape)

        self.params = torch.cat((a,b,c,d,e), dim = 1)
        # print(self.params.shape)
        # exit()
        # gates; i.e. time constants
        c = self.ensure_gate(c)
        d = self.ensure_gate(c*.09)
        b = b*0
        d = 1 + d*0
        e = e*0
        hp_0 = h[:,:self.n_mem]
        hp_1 = h[:,self.n_mem:]
        h0 = hp_0*(1 -c) + c*self.nl((a + 
                                        b*hp_0**2 - 
                                        hp_1)*hp_0 
                                        + u_emb)
        h1 = hp_1*(1 - d) + d*((e)*hp_0)**4

        return torch.cat((h0, h1), dim = 1)
    

class BI_RNN(nn.Module):
    def __init__(self, n_cell = 1, n_lay = 1, actuator_hidden = 64, 
                 actuator_lay = 4, **kwargs):
        super().__init__()
        self.actuator = MLP(2*n_cell, 1, actuator_lay, actuator_hidden)
        self.decision = nn.Sigmoid()
        self.memory = nn.ModuleList([BI_cell(num_mem = n_cell, **kwargs) for _ in range(n_lay)])
        self.n_cell = n_cell
        self.loss_f = nn.functional.binary_cross_entropy

    def forward(self, u):
        #init memory
        for m in self.memory:
            m.params = None
        mem = .01*torch.randn((u.shape[0], self.n_cell*2))
        mem_f = [mem[0,0].detach()]
        mem_s = [mem[0,1].detach()]
        
        out = torch.zeros_like(u)
        for i in range(u.shape[1]):
            ut = u[:,i][:,None]
            for m in self.memory:
                mem = m(mem, ut)
                out[:,i] = self.decision(self.actuator(mem, out_act = False)).squeeze()
            mem_f.append(mem[0,0].detach())
            mem_s.append(mem[0,1].detach())

        # plt.plot(u[0,:]*.05)
        # plt.plot(mem_f)
        # plt.plot(mem_s)
        # plt.show()
        # plt.clf()
        # print(self.memory[0].a, self.memory[0].b, self.memory[0].c, self.memory[0].d, self.memory[0].e)
        return out
    
    def loss(self, u, target):
        loss = 0
        out = self(u)
        enforce_eval = 1
        # eps = 1e-5
        # z_val = (target == 0).sum(dim = 1)
        # o_val = (target == 1).sum(dim = 1)
        # w = o_val/(z_val + eps)
        loss = self.loss_f(out, target)
        # for i in range(out.shape[1]):

        #     with torch.autograd.set_detect_anomaly(True):
        #         if i%enforce_eval == 0:
        #             loss += self.loss_f(out[:,i], target[:,i])
        #             # print(loss)
            
        #         mask = out[:,i] == 1
        #         if mask.sum() != 0:
        #             loss += self.loss_f(out[:,i][mask], target[:,i][mask])
        #             # print(loss)
        
        return loss



if __name__ == '__main__':
    bir = BI_RNN(actuator_lay=3, n_cell = 1, lay = 3)
    # print(bir)
    o,l = bir(torch.zeros((3,3_000)), target_response = torch.ones((3,3_000)))
    print(o,l)