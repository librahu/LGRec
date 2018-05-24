import numpy as np
import heapq

path = '../data/ml-100k'

neighbor_size = 100
embedding_dim = 64
user_num = 0#943
item_num = 0#1683
#user_map_item = {}
#item_map_user = {}
#with open(path + '.train.rating') as infile:
#	for line in infile.readlines():
#		arr = line.strip().split('\t')
#		u = int(arr[0])
#		v = int(arr[1])
#		user_num = max(user_num, u)
#		item_num = max(item_num, v)
#		if u not in user_map_item:
#			user_map_item[u] = {}
#		if v not in item_map_user:
#			item_map_user[v] = {}
#		user_map_item[u][v] = 1
#		item_map_user[v][u] = 1

#print 'user_num = %d, item_num = %d' % (user_num, item_num)
user_map_item = {}
item_map_user = {}

with open(path + '.train.rating') as infile:
    for line in infile.readlines():
        arr = line.strip().split('\t')
        u, m = int(arr[0]), int(arr[1])
        if u not in user_map_item:
            user_map_item[u] = {}
        if m not in item_map_user:
            item_map_user[m] = {}

        user_num = max(user_num, u)
        item_num = max(item_num, m)

        user_map_item[u][m] = 1
        item_map_user[m][u] = 1

print 'user_num = %d, item_num = %d' % (user_num, item_num)

user_embedding = np.zeros((user_num + 1, embedding_dim))
item_embedding = np.zeros((item_num + 1, embedding_dim))
with open(path + '.factors_U') as infile:
	for line in infile.readlines():
		arr = line.strip().split(' ')
		i = int(arr[0]) + 1
		for j in range(len(arr) - 1):
			user_embedding[i][j] = float(arr[j + 1])
with open(path + '.factors_V') as infile:
	for line in infile.readlines():
		arr = line.strip().split(' ')
		i = int(arr[0]) + 1
		for j in range(len(arr) - 1):
			item_embedding[i][j] = float(arr[j + 1])

with open(path + '.user_neighbor_' + str(neighbor_size) + '.new', 'w') as outfile:
	for u in user_map_item:
		item_map_sim = {}
		for v in user_map_item[u]:
			item_map_sim[v] = user_embedding[u].dot(item_embedding[v])
		ranklist = heapq.nlargest(neighbor_size, item_map_sim, key = item_map_sim.get)
		for v in ranklist:
			outfile.write(str(u) + '\t' + str(v) + '\n')


with open(path + '.item_neighbor_' + str(neighbor_size) + '.new', 'w') as outfile:
	for v in item_map_user:
		user_map_sim = {}
		for u in item_map_user[v]:
			user_map_sim[u] = item_embedding[v].dot(user_embedding[u])
		ranklist = heapq.nlargest(neighbor_size, user_map_sim, key = user_map_sim.get)
		for u in ranklist:
			outfile.write(str(v) + '\t' + str(u) + '\n')


