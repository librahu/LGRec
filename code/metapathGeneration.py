import time
import numpy as np

class Config:
	def __init__(self):
		self.path = '../data/ml-100k'
		self.uafile = self.path + '.ua'
		self.uofile = self.path + '.uo'
		self.umfile = self.path + '.train.rating'
		self.mtfile = self.path + '.mt'
		self.metapaths = ['umum', 'uaum', 'uoum', 'umtm']

class MetapathGeneration:
	def __init__(self, config):
		path = config.path
		uafile = config.uafile
		uofile = config.uofile
		umfile = config.umfile
		mtfile = config.mtfile
		metapaths = config.metapaths


		self.load_um(umfile)
		self.load_ua(uafile)
		self.load_uo(uofile)
		self.load_mt(mtfile)

		for metapath in metapaths:
			outfile = path + '.mp.' + metapath
			if metapath == 'um':
				self.get_um(outfile)
			elif metapath == 'umum':
				self.get_umum(outfile)
			elif metapath == 'uaum':
				self.get_uaum(outfile)
			elif metapath == 'uoum':
				self.get_uoum(outfile)
			elif metapath == 'umtm':
				self.get_umtm(outfile)
			else:
				print 'Wrong Metapath'

	def get_um(self, outfile):
		print 'Get UM...'
		#print outfile
		start_time = time.time()

		n, m = self.um_matrix.shape
		print n, m
		ctn = 0
		with open(outfile, 'w') as outfile:
			for i in range(1, n):
				for j in range(1, m):
					if self.um_matrix[i][j] != 0:
						outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(self.um_matrix[i][j])) + '\n')
						ctn += 1
		print '[%.2f] finished. #nonzero = %d' % (time.time() - start_time, ctn)
		#exit(0)

	def get_uaum(self, outfile):
		print 'Get UAUM'
		start_time = time.time()
		um = self.ua_matrix.dot(self.ua_matrix.T).dot(self.um_matrix)
		n, m = um.shape
		ctn = 0
		with open(outfile, 'w') as outfile:
			for i in range(1, n):
				mean = um[i].mean()
				#tmp_list = []
				for j in range(1, m):
					if um[i][j] > mean:
						outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(um[i][j])) + '\n')
						#tmp_list.append([j, um[i][j]])
						ctn += 1
				#tmp_list.sort(key = lambda x : x[1], reverse = True)
				#num = int(len(tmp_list) * 0.4)
				#ctn += num
				#for j, val in tmp_list[:num]:
				#	outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(val)) + '\n')
		print '[%.2f] finished. #nonzero = %d' % (time.time() - start_time, ctn)

	def get_uoum(self, outfile):
		print 'Get UOUM'
		start_time = time.time()
		um = self.uo_matrix.dot(self.uo_matrix.T).dot(self.um_matrix)
		n, m = um.shape
		ctn = 0
		with open(outfile, 'w') as outfile:
			for i in range(1, n):
				mean = um[i].mean()
				#tmp_list = []
				for j in range(1, m):
					if um[i][j] > mean:
						outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(um[i][j])) + '\n')
						#tmp_list.append([j, um[i][j]])
						ctn += 1
				#tmp_list.sort(key = lambda x : x[1], reverse = True)
				#num = int(len(tmp_list) * 0.4)
				#ctn += num
				#for j, val in tmp_list[:num]:
				#	outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(val)) + '\n')
		print '[%.2f] finished. #nonzero = %d' % (time.time() - start_time, ctn)

	def get_umtm(self, outfile):
		print 'Get UMTM'
		start_time = time.time()
		um = self.um_matrix.dot(self.mt_matrix).dot(self.mt_matrix.T)
		n, m = um.shape
		ctn = 0
		with open(outfile, 'w') as outfile:
			for i in range(1, n):
				mean = um[i].mean()
				#tmp_list = []
				for j in range(1, m):
					if um[i][j] > mean:
						outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(um[i][j])) + '\n')
						#tmp_list.append([j, um[i][j]])
						ctn += 1
				#tmp_list.sort(key = lambda x : x[1], reverse = True)
				#num = int(len(tmp_list) * 0.4)
				#ctn += num
				#for j, val in tmp_list[:num]:
				#	outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(val)) + '\n')
		print '[%.2f] finished. #nonzero = %d' % (time.time() - start_time, ctn)

	def get_umum(self, outfile):
		print 'Get UMUM'
		start_time = time.time()
		um = self.um_matrix.dot(self.um_matrix.T).dot(self.um_matrix)
		n, m = um.shape
		ctn = 0
		with open(outfile, 'w') as outfile:
			for i in range(1, n):
				mean = um[i].mean()
				#tmp_list = []
				for j in range(1, m):
					if um[i][j] > mean:
						outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(um[i][j])) + '\n')
						#tmp_list.append([j, um[i][j]])
						ctn += 1
				#tmp_list.sort(key = lambda x : x[1], reverse = True)
				#num = int(len(tmp_list) * 0.4)
				#ctn += num
				#for j, val in tmp_list[:num]:
				#	outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(val)) + '\n')
		print '[%.2f] finished. #nonzero = %d' % (time.time() - start_time, ctn)



	def load_um(self, umfile):
		unum = 0
		mnum = 0
		with open(umfile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				u, m = int(arr[0]), int(arr[1])
				unum = max(u, unum)
				mnum = max(m, mnum)

		print 'unum = ', unum
		print 'mnum = ', mnum

		self.um_matrix = np.zeros((unum + 1, mnum + 1))
		with open(umfile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				u, m = int(arr[0]), int(arr[1])
				self.um_matrix[u][m] = 1

	def load_ua(self, uafile):
		unum = 0
		anum = 0
		with open(uafile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				u, a = int(arr[0]), int(arr[1])
				unum = max(u, unum)
				anum = max(a, anum)

		print 'unum = ', unum
		print 'anum = ', anum

		self.ua_matrix = np.zeros((unum + 1, anum + 1))
		with open(uafile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				u, a = int(arr[0]), int(arr[1])
				self.ua_matrix[u][a] = 1

	def load_uo(self, uofile):
		unum = 0
		onum = 0
		with open(uofile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				u, o = int(arr[0]), int(arr[1])
				unum = max(u, unum)
				onum = max(o, onum)

		print 'unum = ', unum
		print 'onum = ', onum

		self.uo_matrix = np.zeros((unum + 1, onum + 1))
		with open(uofile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				u, o = int(arr[0]), int(arr[1])
				self.uo_matrix[u][o] = 1


	def load_mt(self, mtfile):
		mnum = 0
		tnum = 0
		with open(mtfile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				m, t = int(arr[0]), int(arr[1])
				mnum = max(m, mnum)
				tnum = max(t, tnum)

		print 'mnum = ', mnum
		print 'tnum = ', tnum

		self.mt_matrix = np.zeros((mnum + 1, tnum + 1))
		with open(mtfile) as infile:
			for line in infile.readlines():
				arr = line.strip().split('\t')
				m, t = int(arr[0]), int(arr[1])
				self.mt_matrix[m][t] = 1

if __name__ == '__main__':
	config = Config()
	metapathGeneration = MetapathGeneration(config)
