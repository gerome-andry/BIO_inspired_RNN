import torch
import numpy as np 
import matplotlib.pyplot as plt

class brc():
    def __init__(self, a, c, k, s):
        self.a = a
        self.c = c
        self.k = k                
        self.s = s         
    
    def next(self, h_t, u_t):
        h_next = torch.zeros_like(h_t)
        h_next[0] = (1-self.c)*h_t[0] + (self.c)*(u_t + (self.a - h_t[1])*h_t[0] + self.k*h_t[0]**3).tanh()
        h_next[1] = self.s*((1.4*h_t[0])**4 - h_t[1]) + h_t[1]

        return h_next
    
if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    tau = .001
    eps = .1
    dt = .1#.001
    A = 1.7
    B = 2.5
    C = .4
    D = .005
    cell = brc(A, C, B, D)
    
    xy = torch.meshgrid(torch.linspace(-1.5,1.5, 100), torch.linspace(-1,2,100), indexing = 'ij')
    xy = torch.cat((xy[0].unsqueeze(0), xy[1].unsqueeze(0)))
    def u_t(t):
        # return .5*(2*torch.pi*.1*t).sin()
        u = torch.zeros_like(t)
        u[torch.bitwise_and(t < 1.2, t>1)] = .5
        # u[torch.bitwise_and(t < 4, t>3)] = -.5
        # u[t > 5] = .2*(t[t>5]-5)
        return u               
    
    htp = cell.next(xy, .5)
    plt.contourf(xy[1], xy[0],htp[0] - xy[0], 0)
    plt.contour(xy[1], xy[0],htp[1] - xy[1], 0)

    htp = cell.next(xy, 0)
    tmax = 15
    n = int(np.ceil(tmax/dt))
    plt.contour(xy[1], xy[0],htp[0] - xy[0], 0)
    plt.contour(xy[1], xy[0],htp[1] - xy[1], 0)
    
    t = torch.linspace(0,tmax,n)
    u = u_t(t)
    h_t = [torch.zeros(2)]
    for ut in u:
        h_t.append(cell.next(h_t[-1], ut))


    plt.plot([h[1] for h in h_t[1:]], [h[0] for h in h_t[1:]], color = 'orange') 

    plt.show()
    plt.plot(t, u)
    plt.plot(t, [h[0] for h in h_t[1:]])
    plt.show()
