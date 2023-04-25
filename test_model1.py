import torch
from config.config import Config
from pytorch_trainer import Trainer, set_gpu, set_seed

from experiment import EXPERIMENT_MAP
    
set_gpu(idx=0, cuda_visible_devices='0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)

kwargs = EXPERIMENT_MAP['task13']
config = Config(config_file='./config/config.ini', **kwargs)
trainer = Trainer(config=config, device=device)


trainer.load_model('/home/int2-user/project_sum/checkpoint/t5-sum-gov/ckpt12.pt')

trainer.test()