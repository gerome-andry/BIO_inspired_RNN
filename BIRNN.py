import torch 
import torch.nn as nn   

def encode_choice(probs):
    # probs of size (B, L, 3)
    choice = torch.argmax(probs, dim = -1)
    choice[choice == 2] = -1 # -> (B,L)
    
    return choice

def decode_choice(choice):
    probs = torch.zeros(choice.shape + (3,))
    
    m0 = (choice == 0)[...,None].repeat(1,1,3)
    m0[...,[1,2]] = False
    probs[m0] = 1

    m1 = (choice == 1)[...,None].repeat(1,1,3)
    m1[...,[0,2]] = False
    probs[m1] = 1

    m2 = (choice == -1)[...,None].repeat(1,1,3)
    m2[...,[0,1]] = False
    probs[m2] = 1

    return probs

class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.head = nn.Linear(in_dim, hidden[0])
        self.layers = nn.ModuleList([nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden)-1)])
        self.tail = nn.Linear(hidden[-1], out_dim)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.head(x)
        for l in self.layers:
            x = x + self.activ(l(x))

        return self.tail(x)
    

class SenseMemAct(nn.Module):
    def __init__(self, sensor_net, actor_net, mem_lay = 1, in_sz = 1, mem_sz = 64, decisions = 3, bias = False):
        super().__init__()
        self.sense = sensor_net
        self.act = actor_net
        self.dec = decisions
        self.mem = [nn.GRU(in_sz, mem_sz, mem_lay, bias = bias, batch_first = True)]
        # self.mem = nn.ModuleList([nn.GRU(in_sz, mem_sz, 1, bias = bias, batch_first = True) for _ in range(mem_lay)])
        self.decision = nn.Softmax(dim = -1)
        self.l = nn.CrossEntropyLoss()

    def forward(self, x): 
        # X of the size (Batch, Sequence_lg, Input_sz)
        # Denoted B,L,N
        B, L, N = x.shape        
        # transfer sequence to sensor -> go back to sequence
        inputs = self.sense(x.reshape((-1, N))).reshape((B,L,-1))
        B, L, N = inputs.shape
        # transfer to memory
        if len(self.mem) > 1:
            memory = []
            mems = []
            for l in range(L):
                for i,m in enumerate(self.mem):
                    if l ==0 and i == 0:
                        hi,_ = m(inputs[:,l,:].unsqueeze(1))
                    else:
                        hi,_ = m(inputs[:,l,:].unsqueeze(1), mems[-1])
                        if i == 0:
                            mems = []

                    mems.append(hi.transpose(0,1))
                memory.append(torch.cat([m.transpose(0,1) for m in mems], dim = 2))

            memory = torch.cat(memory, dim = 1) #retrieve the sequence of hidden states
        else:
            memory,_ = self.mem[0](inputs)

        B, L, M = memory.shape
        # transfer sequence output to actions sequence 
        
        in_act = memory.reshape((-1,3, M//3))
        out = []
        for k in range(3):
            out.append(self.act(in_act[...,k,:]).reshape((B,L,self.dec//3)))

        out = torch.cat(out, dim = -1)
        out = self.decision(out)

        return out, memory.reshap((-1, L, 3, M//3)).mean(dim = -1)

    def loss(self, x, target):
        # X - (B,L,N) | T - (B,L,O), O = 3 (choices)

        mask = (target[:,:,0] != 1)
        pred = self(x)
        
        not_m = torch.bitwise_not(mask)
        # return  self.l(pred[mask], target[mask]) + .5*self.l(pred[not_m], target[not_m])
        return self.l(pred, target)
# probs = torch.rand((1,10,3))
# print(probs)
# ch = encode_choice(probs)
# print(ch)
# print(decode_choice(ch))

