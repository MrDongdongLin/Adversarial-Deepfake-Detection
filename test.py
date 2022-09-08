import random

id_res = ['a', 'b', 'c', 'd', 'e']
print(id_res)

random_seed=10
random.seed(random_seed)
random.shuffle(id_res)
print(id_res)

random.shuffle(id_res)
print(id_res)

id_res.remove(id_res[0])
print(id_res)

