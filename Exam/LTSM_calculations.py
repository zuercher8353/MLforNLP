import torch
import numpy as np

#assumptions all biases are 0
xt = np.array([1, 0, 0, 0])
h_t1= np.array([0, 0, 0])
c_t1 = np.array([0, 0, 0])

Wf = np.array([[1,0,0], [0,1,0], [0,0,1]])
Wi = np.array([[1,0.5,0], [0,1,0.5], [0,0.5,0]])
Wc = np.ones((3,3))
Wo = np.array([[0,1,1], [1,0,1], [1,1,0]])
Vf = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1]])
Vi = np.array([[1,0.5,0,0], [0,1,0.5,0], [0,0.5,0,1]])
Vc = np.ones((3,4))
Vo = np.array([[0,1,1,1], [1,0,1,1], [1,1,0,1]])

c= np.matmul(Wf,h_t1)
d = np.matmul(Vf,xt)
e = torch.tensor(np.matmul(Wf,h_t1) + np.matmul(Vf,xt)).float()

ft = torch.sigmoid(e)
print("ft", ft)

it = torch.sigmoid(torch.tensor(np.matmul(Wi,h_t1) + np.matmul(Vi,xt)).float())
print("it", it)

Ct_hut = torch.tanh(torch.tensor(np.matmul(Wc,h_t1) + np.matmul(Vc,xt)).float() )
print("Ct_hut", Ct_hut)

Ct = (ft * c_t1) + (it * Ct_hut).float()
print("Ct",Ct)

ot = torch.sigmoid(torch.tensor(np.matmul(Wo,h_t1) + np.matmul(Vo,xt)).float())
print("ot", ot)

ht = ot* torch.tanh(Ct) 
print("ht", ht)