import os
import torch
import torch.distributed as dist

# 设置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

def check_nccl():
    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("CUDA is not available.")

    try:
        dist.init_process_group(backend='nccl', init_method='env://')
        print("NCCL is available!")
    except Exception as e:
        print(f"NCCL is not available: {e}")

if __name__ == "__main__":
    check_nccl()


