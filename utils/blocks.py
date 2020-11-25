import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as ini

class StdConv(nn.Module):
    def __init__(self, nin, nout, kern, stri=1, pad=0, bias=True, drop=0.1, relu=True, init="standard"):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kern, stride=stri, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)
        self.relu = relu
        
        if init == "standard":
            pass
        elif init == "xavier":
            self.conv.weight = ini.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        if self.relu:
            return self.drop(F.relu(self.bn(self.conv(x))))
        else:
            return self.drop(self.bn(self.conv(x)))

class StdUpsample(nn.Module):
    def __init__(self, nin, nout, kern, stri, pad=0, drop=None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout,kern, stri, padding=pad)
        self.bn = nn.BatchNorm2d(nout)
        if drop: self.drop = nn.Dropout(drop)
        else: self.drop = None
        
    def forward(self, x):
        if self.drop:
            return self.drop(self.bn(F.relu(self.conv(x))))
        else:
            return self.bn(F.relu(self.conv(x)))

class StdLinear(nn.Module):
    def __init__(self, nin, nout, act='relu'):
        super().__init__()
        self.lin = nn.Linear(nin, nout)
        self.act = act
    def forward(self, x):
        if self.act == 'relu':
            return F.relu(self.lin(x))
        elif self.act == 'tanh':
            return torch.tanh(self.lin(x))
        else:
            return self.lin(x)
    
class DenseBlock(nn.Module):
    def __init__(self, nin, nout, kern=1, stri=1, pad=0):
        super().__init__()
        self.conv1 = StdConv(nin, nin, stri, kern, pad)
        self.conv2 = StdConv(nin, nin, stri, kern, pad)
        self.conv3 = StdConv(nin*2, nin, stri, kern, pad)
        self.conv4 = StdConv(nin*3, nin, stri, kern, pad)
        self.conv5 = StdConv(nin*4, nout, stri, kern, pad)
    
    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o2c = torch.cat([o2,o1], dim=1)
        o3 = self.conv3(o2c)
        o3c = torch.cat([o2c,o3], dim=1)
        o4 = self.conv4(o3c)
        o4c = torch.cat([o3c, o4], dim=1)
        o5 = self.conv5(o4c)
        return o5
    
class DenseLinBlock(nn.Module):
    def __init__(self, nin, nout, relu=True):
        super(DenseLinBlock, self).__init__()
        self.lin1 = StdLinear(nin, nin, relu)
        self.lin2 = StdLinear(nin, nin, relu)
        self.lin3 = StdLinear(nin*2, nin, relu)
        self.lin4 = StdLinear(nin*3, nin, relu)
        self.lin5 = StdLinear(nin*4, nout, relu)
    
    def forward(self, x):
        o1 = self.lin1(x)
        o2 = self.lin2(o1)
        o2c = torch.cat([o2,o1], dim=1)
        o3 = self.lin3(o2c)
        o3c = torch.cat([o2c,o3], dim=1)
        o4 = self.lin4(o3c)
        o4c = torch.cat([o3c, o4], dim=1)
        o5 = self.lin5(o4c)
        return o5
    
class StdConv1d(nn.Module):
    def __init__(self, nin, nout, n_layers=1, kern=3,
                 stri=1, pad=0, groups=1, bias=True, batch_first=True, loud=False):
        super(StdConv1d,self).__init__()
        self.batch_first = batch_first
        self.loud = loud
        layers = []
        in_chan = nin
        if n_layers > 1:
            for i in range(n_layers-1):
                out_chan = in_chan//(i+1)
                layers.append(nn.Conv1d(in_chan, out_chan, kern, stri, pad, groups=groups, bias=bias))
                layers.append(nn.Dropout(.1))
                in_chan = out_chan
                if loud: print(out_chan)
        layers.append(nn.Conv1d(in_chan, nout, kern, stri, pad, bias=bias))
        self.convs = nn.ModuleList(layers)
        
    def forward(self, x):
        if self.batch_first:
            x = x.permute(0,2,1)
            
        for lay in self.convs:
            x = lay(x)
            if self.loud: print(x.shape)
        
        return x
            
            
            
            