import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import random_generator
from utils import extract_time


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")

class Time_GAN_module(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, activation=torch.sigmoid):
        super(Time_GAN_module, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sigma = activation
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
    
            batch_size = x.size(0)

            hidden = self.init_hidden(batch_size).to(device)
            
            
            out, hidden = self.rnn(x, hidden)
            
        
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
           
            
            if self.sigma == nn.Identity:
                idendity = nn.Identity()
                return idendity(out)
                
            out = self.sigma(out)
            
            
            return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
def TimeGAN(data, parameters):
  hidden_dim = parameters["hidden_dim"]
  num_layers = parameters["num_layers"]
  iterations = parameters["iterations"]
  batch_size = parameters["batch_size"]
  epoch = parameters["epoch"]
  no, seq_len, dim = np.asarray(data).shape
  z_dim = dim
  gamma = 1
  data = data.to(device)


  # instantiating every module we're going to train
  Embedder = Time_GAN_module(input_size=z_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=num_layers).to(device)
  Recovery = Time_GAN_module(input_size=hidden_dim, output_size=dim, hidden_dim=hidden_dim, n_layers=num_layers).to(device)
  Generator = Time_GAN_module(input_size=dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=num_layers).to(device)
  Supervisor = Time_GAN_module(input_size=hidden_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=num_layers).to(device)
  Discriminator = Time_GAN_module(input_size=hidden_dim, output_size=1, hidden_dim=hidden_dim, n_layers=num_layers, activation=nn.Identity).to(device)

  # instantiating all optimizers,
  # learning rates chosen through experimentation and comparison with results in the paper
  embedder_optimizer = optim.Adam(Embedder.parameters(), lr=0.0035)
  recovery_optimizer = optim.Adam(Recovery.parameters(), lr=0.01)
  supervisor_optimizer = optim.Adam(Recovery.parameters(), lr=0.001)
  discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=0.01)
  generator_optimizer = optim.Adam(Generator.parameters(), lr=0.01)
  
  # instantiating mse loss & Data Loader
  binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
  MSE_loss = nn.MSELoss()
  loader = DataLoader(data, parameters['batch_size'], shuffle=False)

  random_data = random_generator(batch_size=parameters['batch_size'], z_dim=dim,
                                       T_mb=extract_time(data)[0], max_seq_len=extract_time(data)[1])
  #random_data = torch.tensor(random_data).to(device)
  # Embedding Network Training
  # Here we train embedding & Recovery network jointly
  print('Start Embedding Network Training')
  for e in range(epoch):
    for batch_index, X in enumerate(loader):
        H, _ = Embedder(X.float())
        
        H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

        X_tilde, _ = Recovery(H)
        X_tilde = torch.reshape(X_tilde, (batch_size, seq_len, dim))

        # constants chosen like in the paper
        E_loss0 = 10 * torch.sqrt(MSE_loss(X, X_tilde))

        Embedder.zero_grad()
        Recovery.zero_grad()

        E_loss0.backward(retain_graph=True)

        embedder_optimizer.step()
        recovery_optimizer.step()

        if e in range(1,epoch) and batch_index == 0:
            print('step: '+ str(e) + '/' + str(epoch))
        if e%100 == 0 and batch_index==0:
            print('Embedding Loss:' +str(E_loss0))

  print('Finish Embedding Network Training')


  print('Start Training with Supervised Loss Only')
  for e in range(epoch):
    for batch_index, X in enumerate(loader):

        H, _ = Embedder(X.float())
        H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

        H_hat_supervise, _ = Supervisor(H)
        H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))

        G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])


        Embedder.zero_grad()
        Supervisor.zero_grad()

        G_loss_S.backward(retain_graph=True)

        embedder_optimizer.step()
        supervisor_optimizer.step()

        if e in range(1,epoch) and batch_index == 0:
            print('step: '+ str(e) + '/' + str(epoch))
        if e%100 == 0 and batch_index==0:
            print('Supervisor Loss:' +str(G_loss_S))
  print('Finish Training with Supervised Loss Only')
  
  
  # Joint Training
  print('Start Joint Training')
  for itt in range(epoch):
    for kk in range(2):
      X = next(iter(loader))
      z = random_data
      z = z.float()
        
      e_hat, _ = Generator(z)
      e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))
        
      H_hat, _ = Supervisor(e_hat)
      H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
      Y_fake = Discriminator(H_hat)
      Y_fake = torch.reshape(Y_fake, (batch_size, seq_len, 1))
        
      x_hat, _ = Recovery(H_hat)
      x_hat = torch.reshape(x_hat, (batch_size, seq_len, dim))
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      H_hat_supervise, _ = Supervisor(H)
      H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))

      Generator.zero_grad()
      Supervisor.zero_grad()
      Discriminator.zero_grad()
      Recovery.zero_grad()

      G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])
      binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
      G_loss_U = binary_cross_entropy_loss(Y_fake, torch.ones_like(Y_fake))
        
      G_loss_V1 = torch.mean(torch.abs((torch.std(x_hat, [0], unbiased = False)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
      G_loss_V2 = torch.mean(torch.abs((torch.mean(x_hat, [0]) - (torch.mean(X, [0])))))
      G_loss_V = G_loss_V1 + G_loss_V2
        
      G_loss_S.backward(retain_graph=True)#
      G_loss_U.backward(retain_graph=True)
      G_loss_V.backward(retain_graph=True)#


      generator_optimizer.step()
      supervisor_optimizer.step()
      discriminator_optimizer.step()
      MSE_loss = nn.MSELoss()
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      X_tilde, _ = Recovery(H)
      X_tilde = torch.reshape(X_tilde, (batch_size, seq_len, dim))

      E_loss_T0 = MSE_loss(X, X_tilde)
      E_loss0 = 10 * torch.sqrt(MSE_loss(X, X_tilde))
        
      H_hat_supervise, _ = Supervisor(H)
      H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))

      G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])
      E_loss = E_loss0  + 0.1 * G_loss_S
        
      G_loss_S.backward(retain_graph=True)
      E_loss_T0.backward()
        
      Embedder.zero_grad()
      Recovery.zero_grad()
      Supervisor.zero_grad()
        
      embedder_optimizer.step()
      recovery_optimizer.step()
      supervisor_optimizer.step()
      
    for batch_index, X in enumerate(loader):
      
      z = random_data
      z = z.float()

      H, _ = Embedder(X)
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      Y_real = Discriminator(H)
      Y_real = torch.reshape(Y_real, (batch_size, seq_len, 1))
      
      e_hat, _ = Generator(z)
      e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))

      Y_fake_e = Discriminator(e_hat)
      Y_fake_e = torch.reshape(Y_fake_e, (batch_size, seq_len, 1))
        
      H_hat, _ = Supervisor(e_hat)
      H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
      Y_fake = Discriminator(H_hat)
      Y_fake = torch.reshape(Y_fake, (batch_size, seq_len, 1))
        
      x_hat, _ = Recovery(H_hat)
      x_hat = torch.reshape(x_hat, (batch_size, seq_len, dim))

      Generator.zero_grad()
      Supervisor.zero_grad()
      Discriminator.zero_grad()
      Recovery.zero_grad()
      Embedder.zero_grad()

      # logits first, then targets
      # D_loss_real(Y_real, torch.ones_like(Y_real))
      D_loss_real = nn.BCEWithLogitsLoss()
      DLR = D_loss_real(Y_real, torch.ones_like(Y_real))

      D_loss_fake = nn.BCEWithLogitsLoss()
      DLF = D_loss_fake(Y_fake, torch.zeros_like(Y_fake))

      D_loss_fake_e = nn.BCEWithLogitsLoss()
      DLF_e = D_loss_fake_e(Y_fake_e, torch.zeros_like(Y_fake_e))

      D_loss = DLR + DLF + gamma * DLF_e

      # check discriminator loss before updating
      check_d_loss = D_loss
      if (check_d_loss > 0.15):
        D_loss.backward(retain_graph=True)
        discriminator_optimizer.step()
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))
        
      X_tilde, _ = Recovery(H)
      X_tilde = torch.reshape(X_tilde, (batch_size, seq_len, dim))

      
      z = random_data
      z = z.float()
        
      e_hat, _ = Generator(z)
      e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))
        
      H_hat, _ = Supervisor(e_hat)
      H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
      Y_fake = Discriminator(H_hat)
      Y_fake = torch.reshape(Y_fake, (batch_size, seq_len, 1))
        
      x_hat, _ = Recovery(H_hat)
      x_hat = torch.reshape(x_hat, (batch_size, seq_len, dim))
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      H_hat_supervise, _ = Supervisor(H)
      H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))

      G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])
      binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
      # logits first then targets
      G_loss_U = binary_cross_entropy_loss(Y_fake, torch.ones_like(Y_fake))
        
      G_loss_V1 = torch.mean(torch.abs((torch.std(x_hat, [0], unbiased = False)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
      G_loss_V2 = torch.mean(torch.abs((torch.mean(x_hat, [0]) - (torch.mean(X, [0])))))
      G_loss_V = G_loss_V1 + G_loss_V2
    
      E_loss_T0 = MSE_loss(X, X_tilde)
      E_loss0 = 10 * torch.sqrt(MSE_loss(X, X_tilde))
      E_loss = E_loss0  + 0.1 * G_loss_S
        
      G_loss_S.backward(retain_graph=True)#
      G_loss_U.backward(retain_graph=True)
      G_loss_V.backward(retain_graph=True)#
      E_loss.backward()

      generator_optimizer.step()
      supervisor_optimizer.step()
      embedder_optimizer.step()
      recovery_optimizer.step()

      
      random_test = random_generator(1, dim, extract_time(data)[0], extract_time(data)[1])
      sample_input = random_generator(1, dim, extract_time(data)[0], extract_time(data)[1])
    
      test_sample = Generator(sample_input)[0]
      test_sample = torch.reshape(test_sample, (1, seq_len, hidden_dim))
      test_recovery = Recovery(test_sample)
      test_recovery = torch.reshape(test_recovery[0], (1, seq_len, dim))
      
    print('step: '+ str(itt+1) + '/' + str(epoch))
    if e%100 == 0 and batch_index==0:
     print('Supervisor Loss:' +str(G_loss_S.data) + 'Generator Loss:' +str(G_loss_U.data) + 'Discriminator Loss:' +str(G_loss_V.data)+ 'Embedding Loss:' +str(E_loss.data))             
  print('Finish Joint Training')
                
  return Generator, Embedder, Supervisor, Recovery, Discriminator