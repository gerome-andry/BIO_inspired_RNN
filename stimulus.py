import torch
import numpy as np
import matplotlib.pyplot as plt   
import os
import torch.nn as nn 

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class StimGenerator():
    def __init__(self, rest = 0, max_t = 20, decision_dt = 1, 
                 stim_dt = 2, cooldown_dt = 1, 
                 dt = 1e-4, freq = 1):
        self.rest_val = rest              
        self.dt_inout = decision_dt
        self.dt_stim = stim_dt
        self.cooldown = cooldown_dt
        self.max_t = max_t
        # self.n = n_pulses
        self.dt = dt
        self.f = freq

    def get_batch_data(self, nb = 256):
        times = torch.linspace(0, self.max_t, int(self.max_t//self.dt))
        t_idx = torch.arange(len(times))

        # get time of first signal
        # first stim is fully in the first half of the whole signal
        # first signal must be after 'START_dt' + cooldown
        first_stim_ok = self.dt_inout + self.cooldown
        first_stim_stop = .5*(self.max_t - self.cooldown) - self.dt_stim
        t1_mask = torch.bitwise_and(times > first_stim_ok, times < first_stim_stop)
        t1 = torch.tensor(np.random.choice(t_idx[t1_mask], size = nb, replace = True))

        # get time of second signal
        # second stim is fully in the second half of the whole signal
        # second signal must be before max_t - 'END_dt' - cooldown
        scnd_stim_ok = .5*(self.max_t + self.cooldown)
        scnd_stim_stop = self.max_t - self.cooldown - self.dt_stim - self.dt_inout
        t2_mask = torch.bitwise_and(times > scnd_stim_ok, times < scnd_stim_stop)
        t2 = torch.tensor(np.random.choice(t_idx[t2_mask], size = nb, replace = True))
        
        # determine if signal will be of type fast or slow -> True = fast, False = slow
        types = torch.randint(2, (nb, 2), dtype = torch.bool)
        # compute the expected decision -> True = same, False = different
        decision = types[:,0] == types[:,1]
        
        # create input (START - S1 - S2 - STOP)
        input_s = torch.zeros((nb, len(times))) + .1
        idx_stim = torch.arange(int(self.dt_stim//self.dt))[None,:].expand(nb, -1)
        s1_t = t1[:,None] + idx_stim
        s2_t = t2[:,None] + idx_stim  
        t_start_sign = int(self.dt_inout//self.dt)
        t_stop_sign = len(times) - 1 - t_start_sign
        f_s_b = self.f_stim()
        s_s_b = self.s_stim()
        input_s.scatter_(1, s1_t, f_s_b*(types[:,:1].expand(-1,len(f_s_b))) +
                          s_s_b*(torch.bitwise_not(types[:,:1]).expand(-1,len(f_s_b))))
        input_s.scatter_(1, s2_t, f_s_b*(types[:,1:].expand(-1,len(f_s_b))) +
                          s_s_b*(torch.bitwise_not(types[:,1:]).expand(-1,len(f_s_b))))
        for i, (t1,t2) in enumerate(zip(s1_t[:,-1], s2_t[:,0])):
            input_s[i, t1:t2] = .5

        input_s[:,:t_start_sign] = 1
        input_s[:,t_stop_sign:] = 1
        # create desired output (0 - DECISION), decision is 1 (same), -1 (different)
        output_s = torch.zeros_like(input_s)
        output_s[:,t_stop_sign:] = 1*decision[:,None].expand(-1, len(times) - t_stop_sign)
        output_s[:,t_stop_sign:] -= 1*torch.bitwise_not(decision)[:,None].expand(-1, len(times) - t_stop_sign)

        return input_s + self.rest_val, output_s

    def f_stim(self):
        t = torch.linspace(0, self.dt_stim, int(self.dt_stim//self.dt))
        # stim = (3*torch.pi*self.f*t).sin()
        stim = t*0 + .3
        return nn.functional.relu(stim)
    
    def s_stim(self):
        t = torch.linspace(0, self.dt_stim, int(self.dt_stim//self.dt))
        stim = (torch.pi*self.f*t).sin()
        stim = t*0 + .7

        return nn.functional.relu(stim)

    
    def extend_sim(self, time_int, i, o):
        lg = int(time_int//self.dt)
        b, l_stim = i.shape
        new_i = torch.zeros((b, lg)) + self.rest_val
        new_o = torch.zeros_like(new_i)
        new_times = torch.linspace(0, time_int, lg)
        t_idx = torch.arange(len(new_times))

        stim_ok = 0
        stim_stop = time_int - self.max_t - self.cooldown
        t_mask = torch.bitwise_and(new_times > stim_ok, new_times < stim_stop)
        t = torch.tensor(np.random.choice(t_idx[t_mask], size = b, replace = True))
        idx_stim = torch.arange(l_stim)
        insert_stim = t[:,None] + idx_stim

        new_i.scatter_(1, insert_stim, i)
        new_o.scatter_(1, insert_stim, o)
        l_add = int(self.cooldown//self.dt)
        insert_cd = insert_stim[:,-1:] + torch.arange(l_add)
        new_o.scatter_(1, insert_cd, o[:,-1:].expand(-1, l_add))

        return new_i, new_o
    
    def concat_sim(self, i, o):
        # add_cooldown = torch.zeros(int(self.cooldown//self.dt))
        # l_add = len(add_cooldown)
        # b,l = i.shape
        # new_i = torch.zeros((b,l + l_add))
        # new_o = torch.zeros_like(new_i)
        # new_i[:,:l] = i
        # new_o[:,:l] = o

        return i.view(1,-1), o.view(1,-1)

        
if __name__ == '__main__':
    sg = StimGenerator(dt = .01, rest = 0)
    B = 10
    i,o = sg.get_batch_data(B)
    i,o = sg.extend_sim(30, i, o)
    # print(i.shape)

    for k in range(int(B//2)):
        ii,oo = sg.concat_sim(i[2*k:2*(k+1),:],o[2*k:2*(k+1),:])
        plt.plot(ii.squeeze())
        plt.plot(oo.squeeze())
        plt.show(block = True)
        plt.pause(.1)
        plt.clf()