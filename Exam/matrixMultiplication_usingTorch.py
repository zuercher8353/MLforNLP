import torch
'''
bmm: batch matrix multiplication
variable   dim
    x:     b*n*m
    y:     b*m*p
z=bmm(x,y): b*n*p     

transpose(dim0, dim1) == transpose(dim1, dim0)
=> dim0 and dim1 get swap 

'''
a=torch.randn(3, 3, 5)
b=torch.randn(3, 5, 13)
c=torch.randn(3, 3, 13) # fill the blanks
x=torch.bmm(a,b)
y=torch.bmm(x,c.transpose(2,1)) 
print(y.shape)