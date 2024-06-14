import os

import torch
import numpy as np
import random
def set_seed(seed):

   random.seed(seed)  # 为python设置随机种子
   np.random.seed(seed)  # 为numpy设置随机种子
   torch.manual_seed(seed)  # 为CPU设置随机种子
   torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
   # torch.use_deterministic_algorithms(True)

   # if seed!=None:
   #    torch.manual_seed(seed)
   #    torch.cuda.manual_seed(seed)
   #    np.random.seed(seed)
   #    random.seed(seed)
   #    torch.backends.cudnn.deterministic = True
   # torch.backends.cudnn.benchmark = True
   # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
   # torch.use_deterministic_algorithms(True)