import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# batch size
BATCH_SIZE = 32

# 折扣率
GAMMA = 0.99

# 开始更新参数的步数 size=10000
INITIAL_SIZE = 10000

# 经验池容量
CAPACITY = 10 * INITIAL_SIZE

# 学习率
lr = 1e-4

# 目标网络更新一次的步数
TARGET_UPDATE = 1000

# 训练次数
N_EPISODES = 1000