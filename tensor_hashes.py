import torch

def tensor_hash(x:torch.tensor, quant_scale = 0.05)-> torch.tensor:
  """ hash a multi-dim tensor for unit test verification """
  quant_tensor = torch.quantize_per_tensor(x,quant_scale,0, torch.qint32)
  hashed_tensor = quant_tensor.int_repr().sum(-1) % 10
  
  return hashed_tensor
