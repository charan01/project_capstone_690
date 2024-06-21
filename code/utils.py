import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation. (Generates a latent vector that is used in the Generator.)

    Args:
        - batch_size: size of the random vector
        - z_dim: dimension of random vector
        - T_mb: time information for the random vector
        - max_seq_len: maximum sequence length

    Returns:
        - Z_mb: generated random tensor
    """
    Z_mb = []
    for i in range(batch_size):
        temp = torch.zeros(max_seq_len, z_dim)
        temp_Z = torch.FloatTensor(T_mb[i], z_dim).uniform_(0., 10.)
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    return torch.stack(Z_mb, dim=0).to(device)  # Stack tensors along a new dimension (batch dimension)

    
def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len

