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


class nBRC(nn.Module): #extend to multiple layers ?
    def __init__(self, in_sz, mem_sz, mem_lay = 1, bias = False, batch_first = True):
        super().__init__()

        self.ff_im = nn.Linear(in_sz, 3*mem_sz, bias = bias)
        self.ff_mm = nn.Linear(mem_sz, 2*mem_sz, bias = bias)

        self.mem_sz = mem_sz

    def step(self, x, h, bist = False): #x of the form (B,N), h -> (B,M)
        in_emb = self.ff_im(x)
        i_a, i_c, i_o = in_emb.split(self.mem_sz, 1)
        mem_emb = self.ff_mm(h)
        m_a, m_c = mem_emb.split(self.mem_sz, 1)

        a = 1 + torch.tanh(i_a + m_a)
        c = torch.sigmoid(i_c + m_c)

        hfn = c*h + (1-c)*torch.tanh(i_o + a*h)
        if bist:
            return a,c, hfn
        
        return hfn
    
    def forward(self, u, h0 = None, mem = False): #u -> (B,L,N), h0 initial mem (B,M)
        B, L, _ = u.shape
        if h0 is None:
            h0 = torch.zeros((B, self.mem_sz)).to(u)

        h_t = [h0]
        if mem:
            al,cl = [],[]
        for i in range(L):
            if mem:
                a,c,h_next= self.step(u[:,i], h_t[-1], bist = True)
                al.append(a)
                cl.append(c)
            else:
                h_next = self.step(u[:,i], h_t[-1])
            h_t.append(h_next)
        
        h_t = [h.unsqueeze(1) for h in h_t[1:]]

        if mem:
            a_t = [at.unsqueeze(1) for at in al]
            c_t = [ct.unsqueeze(1) for ct in cl]
            
            return torch.cat(a_t, dim = 1),\
                   torch.cat(c_t, dim = 1),\
                   torch.cat(h_t, dim = 1)
        
        return torch.cat(h_t, dim = 1), h_t[-1] # (B,L,M)

class nBEFRC(nn.Module): #extend to multiple layers ?
    def __init__(self, in_sz, mem_sz, mem_lay = 1, bias = False, batch_first = True, dt = .1):
        super().__init__()
        self.ff_im = nn.Linear(in_sz, 6*mem_sz)
        self.ff_mm = nn.Linear(mem_sz, 5*mem_sz)

        self.mem_sz = mem_sz
        self.dt = dt

    def step(self, x, hf, hs, bist = False): #x of the form (B,N), h -> (B,M)
        in_emb = self.ff_im(x)
        i_a, i_b, i_c, i_d, i_e, i_o = in_emb.split(self.mem_sz, 1)
        mem_emb = self.ff_mm(hf)
        m_a, m_b, m_c, m_d, m_e = mem_emb.split(self.mem_sz, 1)

        a = 1 + torch.tanh(i_a + m_a)
        b = (3/2)*(1 + torch.tanh(i_b + m_b)) #b in [0, 3]
        c = 3*self.dt + (1-3*self.dt)*torch.sigmoid(i_c + m_c) #fast 1/tau in [3, 10]
        d = .3*self.dt*torch.sigmoid(i_d + m_d) #slow epsilon in [0, .3] (one order below the fast)
        e = 1 + torch.sigmoid(i_e + m_e) #e in [1,2]

        hfn = (1-c)*hf + c*torch.tanh(i_o + (a + b*hf**2 - hs)*hf)
        hsn = hs*(1-d) + d*(e*hf)**4
        if bist:
            return a,b,c,d,e, hfn, hsn
        
        return hfn, hsn
                
    
    def forward(self, u, h0 = None, mem = False): #u -> (B,L,N), h0 initial mem (B,M)
        B, L, _ = u.shape
        if h0 is None:
            h0 = torch.zeros((2, B, self.mem_sz)).to(u)

        hf_t = [h0[0]]
        hs_t = [h0[1]]
        if mem:
            al,bl,cl,dl,el = [],[],[],[],[]
        for i in range(L):
            if mem:
                a,b,c,d,e,hf_next, hs_next = self.step(u[:,i], hf_t[-1], hs_t[-1], bist = True)
                al.append(a)
                bl.append(b)
                cl.append(c)
                dl.append(d)
                el.append(e)
            else:
                hf_next, hs_next = self.step(u[:,i], hf_t[-1], hs_t[-1])
            hf_t.append(hf_next)
            hs_t.append(hs_next)
        
        h_t = [h.unsqueeze(1) for h in hf_t[1:]]
        h_s = [h.unsqueeze(1) for h in hs_t[1:]]
        if mem:
            a_t = [at.unsqueeze(1) for at in al]
            b_t = [bt.unsqueeze(1) for bt in bl]
            c_t = [ct.unsqueeze(1) for ct in cl]
            d_t = [dt.unsqueeze(1) for dt in dl]
            e_t = [et.unsqueeze(1) for et in el]
            
            return torch.cat(a_t, dim = 1),\
                   torch.cat(b_t, dim = 1),\
                   torch.cat(c_t, dim = 1),\
                   torch.cat(d_t, dim = 1),\
                   torch.cat(e_t, dim = 1),\
                   torch.cat(h_t, dim = 1),\
                   torch.cat(h_s, dim = 1),
        
        return torch.cat(h_t, dim = 1), h_t[-1] # (B,L,M)
    
class _surrogate_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        ctx.save_for_backward(h)
        return nn.functional.relu(h)
    
    @staticmethod
    def backward(ctx, grad_output):
        h, = ctx.saved_tensors
        return grad_output, None
    
class mSRC(nn.Module):
    def __init__(self, in_sz, mem_sz, mem_lay = 1, bias = False, batch_first = True, dt = .1):
        super().__init__()
        self.ff_im = nn.Linear(in_sz, mem_sz*2)
        self.ff_mm = nn.Linear(mem_sz, mem_sz)

        self.mem_sz = mem_sz
        self.dt = dt
        self.soma_fast = 4
        self.soma_slow = 7
        self.modulation_bias = 0.4
        #self.soma_bias = 1
        self.soma_eps_base = 0.9
        self.bias_mod_range = 5
        self.activation = _surrogate_relu.apply
    def variable_eps(self,fast_potentials):
        return self.soma_eps_base + self.soma_eps_base / \
                (1 + torch.exp((fast_potentials - .5) / -.1))

    def step(self, x, hf, hs, bist = False): #x of the form (B,N), h -> (B,M)
        in_emb = self.ff_im(x)
        soma_bias_contribution_1, i_o = in_emb.split(self.mem_sz, 1)


        soma_bias = torch.tanh(soma_bias_contribution_1 /self.bias_mod_range)*self.bias_mod_range
        hfn = torch.tanh(
                i_o
                + self.soma_fast * hf
                - self.soma_slow * torch.square(hs + self.modulation_bias)
                + soma_bias)
        #hfn = self.activation(hfn)

        slow_eps = self.variable_eps(hf)

        hsn = (1-slow_eps)*hs + slow_eps*hf

        if bist:
            return soma_bias, hfn, hsn
        
        return hfn, hsn
                
    
    def forward(self, u, h0 = None, mem = False): #u -> (B,L,N), h0 initial mem (B,M)
        B, L, _ = u.shape
        if h0 is None:
            h0 = torch.zeros((2, B, self.mem_sz)).to(u)

        hf_t = [h0[0]]
        hs_t = [h0[1]]
        if mem:
            current_bias = []
        for i in range(L):
            if mem:
                b1,hf_next, hs_next = self.step(u[:,i], hf_t[-1], hs_t[-1], bist = True)
                current_bias.append(b1)
            else:
                hf_next, hs_next = self.step(u[:,i], hf_t[-1], hs_t[-1])
            hf_t.append(hf_next)
            hs_t.append(hs_next)
        
        h_t = [h.unsqueeze(1) for h in hf_t[1:]]
        h_s = [h.unsqueeze(1) for h in hs_t[1:]]
        if mem:
            b1_t = [at.unsqueeze(1) for at in current_bias]

            return torch.cat(b1_t, dim = 1),\
                   torch.cat(h_t, dim = 1),\
                   torch.cat(h_s, dim = 1),
        
        return torch.cat(h_t, dim = 1), h_t[-1] # (B,L,M)

class SenseMemAct(nn.Module):
    def __init__(self, sensor_net, actor_net, type = 'BRC', mem_lay = 1, in_sz = 1, mem_sz = 64, decisions = 3, bias = False, ortho = False):
        super().__init__()
        self.sense = sensor_net
        self.act = actor_net
        self.dec = decisions
        self.memsz = mem_sz
        self.orth = ortho
        self.type = type
        if type == 'BRC':
            self.mem = nBRC(in_sz, mem_sz, mem_lay, bias = bias, batch_first = True)
        elif type == 'BEF':
            self.mem = nBEFRC(in_sz, mem_sz, mem_lay, bias = bias, batch_first = True)
        elif type == 'GRU':
            self.mem = nn.GRU(in_sz, mem_sz, mem_lay, bias = bias, batch_first = True)
        elif type == 'mSRC':
            self.mem = mSRC(in_sz, mem_sz, mem_lay, bias = bias, batch_first = True)
        else:
            raise NotImplementedError()
        # self.mem = nn.ModuleList([nn.GRU(in_sz, mem_sz, 1, bias = bias, batch_first = True) for _ in range(mem_lay)])
        self.decision = nn.Softmax(dim = -1)
        self.l = nn.CrossEntropyLoss()

    def forward(self, x, debug_mem = False, mem = False): 
        # print(x, debug_mem)
        # X of the size (Batch, Sequence_lg, Input_sz) 
        # Denoted B,L,N
        with torch.no_grad():
            # orthogonal matrix hidden-hidden
            if self.orth:
                u,_,v = torch.linalg.svd(self.mem[0].weight_hh_l0[:self.memsz,:])
                self.mem[0].weight_hh_l0[:self.memsz,:] = u@v
        
        B, L, N = x.shape        
        # transfer sequence to sensor -> go back to sequence
        inputs = self.sense(x.reshape((-1, N))).reshape((B,L,-1))
        # transfer to memory
        if not debug_mem:
            memory,_ = self.mem(inputs)
            B, L, M = memory.shape
            # transfer sequence output to actions sequence 
            out = self.decision(self.act(memory.reshape((-1, M))).reshape((B,L,self.dec)))
            if mem:
                out = (out, memory)
        
        else:
            if self.type != 'GRU':
                out = self.mem(inputs, mem = True)

            else:
                out,_ = self.mem(inputs)

        return out

    def loss(self, x, target):
        # X - (B,L,N) | T - (B,L,O), O = 3 (choices)

        pred, mem = self(x, mem = True)

        mask = (target[:,:,0] != 1)
        not_m = torch.bitwise_not(mask)
        pred_dec, targ_dec = pred[mask], target[mask]
        pred_ndec, targ_ndec = pred[not_m], target[not_m]

        #compute correlation
        corr = []
        for m in mem:
            corr.append(-torch.corrcoef(m.T)[0].square().sum())

        corr = torch.tensor(corr).mean()

        return  self.l(pred_dec, targ_dec) + self.l(pred_ndec, targ_ndec)# + corr/self.memsz
        # return self.l(pred, target.transpose(-2,-1))


