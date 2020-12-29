import torch
'''
squeeze(x):
 - removes dim of size 1 at idx x
 - if not 1 at idx x do nothing

unsqueeze(x)
 - insert a dim of size at pos x
    e.g.                                        
    x = torch.randn(3,2,5)
    x = unsqueeze(x, 2) => shape [3,2,1,5]
'''

x=torch.randn(3, 2, 5)
x=torch.unsqueeze(x, 3)
print(x.shape)

x=torch.squeeze(x, 0)
print(x.shape)


