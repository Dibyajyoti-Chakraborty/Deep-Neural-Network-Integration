#Copyright (c) 2022-2082, D. Chakraborty and S. Gopalakrishnan
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import torch
import torch.autograd as autograd         # for automatic differentiation              
import torch.nn as nn                     
torch.set_default_tensor_type(torch.DoubleTensor)
from tqdm import tqdm

class DNNI(nn.Module):
    
    def __init__(self,layers,device):
        super().__init__() 
        self.device = device
        self.layers = layers      
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]).to(device)
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
            
    def forward(self,x):
            if torch.is_tensor(x) !=True:
                x= torch.from_numpy(x).to(self.device)
            sigma = x.type(torch.DoubleTensor).to(self.device)
            for i in range(len(self.layers)-2):
                z = self.linears[i](sigma)
                sigma = self.activation(z)
            sigma = self.linears[-1](sigma)
            return sigma
    
    #Modify the loss function as per the problem
    def loss_func(self, x_train,int_f):        
        #x = x_train[:,[0]]
        #a = torch.zeros(x_train.shape[1]-1)
        #for i in range(1,x_train.shape[1]):
        #    a[i] = x_train[:,[i]]               
        g = x_train.clone()
                        
        g.requires_grad = True
        
        u = self.forward(g)
                
        u_x_others = autograd.grad(u,g,torch.ones([x_train.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
                                                            
        u_x = u_x_others[:,[0]]
        
        #loss_f = self.loss_function(u_x,int_f(x,a))
        loss_f = self.loss_function(u_x,int_f(x_train))
                
        return loss_f
                                           
    def closure(self,x_train,int_f,steps,eps=1e-8,lr=5e-2,show=True):
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        for i in tqdm(range(steps)):
            loss = self.loss_func(x_train,int_f)
            self.mse = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                #Learning rate scheduling. It performs better using this even for Adam.
            if i%(steps/4)==0:
                lr=lr/5
                optimizer = torch.optim.Adam(self.parameters(),lr=lr)
                if show==True:
                    with torch.no_grad():
                        print('Iter: ',i,'Loss: ',loss.detach().cpu().numpy(),' lr: ',lr)
            if self.mse<=eps:
                break
        if show==True: print('MSE : ',loss.detach().cpu().numpy())        
            
           
