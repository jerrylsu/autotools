import numpy as np

random_order = list(range(len(dataset)))
np.random.shuffle(random_order)

# 划分valid
train_data = [dataset[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [dataset[j] for i, j in enumerate(random_order) if i % 10 == 0]
