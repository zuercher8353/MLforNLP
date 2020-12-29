import torch
from torch import nn

#assumption classififaction task with 2 classes, represented by label 0 or 1
tensor = torch.rand(1,4,3, requires_grad=True)
#labels 0,1 
labels = torch.randint(low= 0, high= 2, size=(1,4,1))

#should we do this 
#ffn = nn.Linear(3,1)

#or this?
class ClassificationNet(nn.Module):

    def __init__(self):
        super(ClassificationNet, self).__init__()
    
        self.fc1 = nn.Linear(3, 1)


    def forward(self, x):
        
        #TODO 2
        x = torch.sigmoid(self.fc1(x))
        return x
#creat net
ffn = ClassificationNet()
#creat loss function
criterion = nn.BCELoss()
#calculate output
output = ffn(tensor)
#calculate the loss
loss = criterion(output, labels.float())

print(loss)
