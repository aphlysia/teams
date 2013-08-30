#!/usr/bin/env python
import numpy as np
import math
import random

def getColumns(rows, indices, fs = None, printError = False):
	l = len(indices)
	xs = [[] for _ in range(l)]
	_xs = [1] * l
	errorRows = []
	if fs is None:
		fs = ((lambda x: x),) * l
	for row in rows:
		try:
			for i in range(l):
				_xs[i] = fs[i](row[indices[i]])
			for i in range(l):
				xs[i].append(_xs[i])
		except Exception as ex:
			if printError:
				print(ex)
				print('\t' + str(row))
			errorRows.append(row)
	return xs, errorRows

def count(xs):
	l = len(xs)
	c = {}
	for i in range(len(xs[0])):
		key = tuple([xs[j][i] for j in range(l)])
		c[key] = c.get(key, 0) + 1
	return c

def split(x, y, isClass0):
	'''
	split x into class0 and class1
	'''
	x = np.matrix(x)
	r, c = x.shape
	x0 = None
	x1 = None
	for i in range(c):
		if isClass0(y[0,i]):
			x0 = x[:,i] if x0 is None else np.hstack((x0, x[:,i]))
		else:
			x1 = x[:,i] if x1 is None else np.hstack((x1, x[:,i]))
	return x0, x1

def mean(x):
	'''
	x[i,j] is the i-th elements of the x at the j trial
	'''
	return np.matrix(x).mean(1)

def covariance(x):
	'''
	x[i,j] is the i-th elements of the x at the j trial
	'''
	return np.matrix(np.cov(x))

class Z:
	def __init__(self, x = None):
		if x is not None:
			self.train(x)
	def train(self, x):
		self.m = np.array(mean(x))
		self.v = np.array(covariance(x).diagonal().T) ** 0.5
	def z(self, x):
		x = np.array(x)
		return np.matrix((x - self.m) / self.v)
	def inv(self, z):
		z = np.array(z)
		return np.matrix(z * self.v + self.m)

def MahalanobisD(x, mu, SI):
	_x = x - mu
	return (_x.T * SI * _x)[0,0]
	
class Discriminator:
	def __init__(self, x0 = None, x1 = None):
		if x0 is not None and x1 is not None:
			self.train(x0, x1)
	def train(self, x0, x1):
		raise NotImprementedError
	def do(self, x):
		raise NotImprementedError

class Fisher(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		S0 = covariance(x0)
		S1 = covariance(x1)
		_, n0 = x0.shape
		_, n1 = x1.shape
		S = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
		self.SI = S.I
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI)
		d1 = MahalanobisD(x, self.mean1, self.SI)
		return 0 if d0 < d1 else 1
		
class Quadratic(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		self.SI0 = covariance(x0).I
		self.SI1 = covariance(x1).I
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI0)
		d1 = MahalanobisD(x, self.mean1, self.SI1)
		return 0 if d0 < d1 else 1

class Gauss(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		S0 = covariance(x0)
		self.SI0 = S0.I
		self.d0 = np.linalg.det(S0) ** 0.5
		S1 = covariance(x1)
		self.SI1 = S1.I
		self.d1 = np.linalg.det(S1) ** 0.5
		dim, _ = x0.shape
		self.k = (2. * math.pi)**(-dim / 2.)
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI0)
		d1 = MahalanobisD(x, self.mean1, self.SI1)
		p0 = self.k * math.exp(-d0 / 0.5) / self.d0
		p1 = self.k * math.exp(-d1 / 0.5) / self.d1
		return 0 if p0 > p1 else 1

class BayseGauss(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		S0 = covariance(x0)
		self.SI0 = S0.I
		self.d0 = np.linalg.det(S0) ** 0.5
		S1 = covariance(x1)
		self.SI1 = S1.I
		self.d1 = np.linalg.det(S1) ** 0.5
		dim, _ = x0.shape
		self.k = (2. * math.pi)**(-dim / 2.)
		_, c0 = x0.shape
		_, c1 = x1.shape
		self.p0 = float(c0) / (c0 + c1)
		self.p1 = 1. - self.p0
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI0)
		d1 = MahalanobisD(x, self.mean1, self.SI1)
		p0 = self.k * math.exp(-d0 / 0.5) / self.d0 * self.p0
		p1 = self.k * math.exp(-d1 / 0.5) / self.d1 * self.p1
		return 0 if p0 > p1 else 1

def crossValidation(x0, x1, discriminator, bin = 1):
	success = 0
	failure = 0
	r, c0 = x0.shape
	r, c1 = x1.shape
	l = [0] * c0 + [1] * c1
	random.shuffle(l)
	l0 = 0
	l1 = 0
	for i in range(0, c0 + c1, bin):
		r0 = l0 + l[i:i+bin].count(0)
		r1 = l1 + l[i:i+bin].count(1)
		_x0 = np.hstack((x0[:,:l0], x0[:,r0:]))
		_x1 = np.hstack((x1[:,:l1], x1[:,r1:]))
		d = discriminator(_x0, _x1)
		for j in range(l0, r0):
			if d.do(x0[:,j]) == 0:
				success += 1
			else:
				failure += 1
		for j in range(l1, r1):
			if d.do(x1[:,j]) == 1:
				success += 1
			else:
				failure += 1
		l0 = r0
		l1 = r1
	return success, failure 

class PCA:
	def __init__(self, x = None):
		if x is not None:
			self.train(x)
	def train(self, x):
		self.mean = mean(x)
		cov = covariance(x)
		w, v = np.linalg.eig(cov)
		self.eigenvalues = w
		self.eigenvectors = v.T
	def do(self, x):
		return self.eigenvectors * (x - self.mean)
	def plot(self, x, c = 'b'):
		import matplotlib.pyplot as plt
		v = self.do(x)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(list(np.array(v[0])[0]), list(np.array(v[1])[0]), c = c)
		plt.show()

class Cluster:
	def __init__(self, x = None, k = None):
		if x is not None:
			self.train(x, k)
	def train(self, x, k = None):
		# k is the number of clusters if needed
		raise NotImprementedError
	def classify(self, x):
		raise NotImprementedError
	def groups(self, x):
		m, n = x.shape
		classes = self.classify(x)
		d = {}
		count = {}
		for i in set(classes):
			d[i] = np.matrix(np.zeros((m, classes.count(i))))
			count[i] = 0
		for i in range(n):
			c = classes[i]
			d[classes[i]][:,count[c]] = x[:,i]
			count[c] += 1
		return d

class KMeans(Cluster):
	def classify(self, x):
		m, n = x.shape
		classes = [0] * n
		for i in range(n):
			d = np.array(self.y - x[:,i])
			distances = list(np.ones((1,m)).dot(d * d)[0])
			classes[i] = distances.index(min(distances))
		return classes

	def train(self, x, k):
		m, n = x.shape
		def centers(x, classes):
			y = np.matrix(np.zeros((m, k)))
			for i in range(k):
				b = np.matrix(np.zeros((n, 1)))
				for j in range(n):
					if classes[j] == i:
						b[j, 0] += 1. / classes.count(i)
				y[:,i] = x * b
			return y
		
		#initialize
		y = np.matrix(np.zeros((m, k)))
		for i in range(k):
			y[:,i] = x[:,random.randrange(n)]
		self.y = y

		classes = self.classify(x)
		self.y = centers(x, classes)
		new_classes = self.classify(x)
		while classes != new_classes:
			classes = new_classes
			self.y = centers(x, classes)
			new_classes = self.classify(x)

class Gaussian:
	def __init__(self, mu, S):
		m, n = mu.shape
		self.m = m
		self.k = 1. / (((2. * math.pi) ** (m / 2.)) * (np.linalg.det(S) ** 0.5))
		self.mu = mu
		self.L = np.linalg.cholesky(S)
		self.SI = S.I
	def p(self, x):
		d = MahalanobisD(x, self.mu, self.SI)
		return self.k * math.exp(-d / 2.)
	def sample(self):
		x = np.matrix(np.zeros((self.m, 1), dtype=float))
		for i in range(self.m):
			x[i,0] = random.gauss(0,1)
		return self.mu + self.L * x

class DiscreteDistribution:
	def __init__(self, pis):
		m = len(pis)
		self.pis = np.zeros(m)
		s = 0
		for i in range(m):
			s += pis[i]
			self.pis[i] = s
		self.pis /= self.pis[-1]
	def sample(self):
		return list(self.pis > random.random()).index(True)

class GaussianMixture:
	def __init__(self, mus, Ss, pis):
		self.gs = []
		self.k = len(mus)
		for i in range(self.k):
			self.gs.append(Gaussian(mus[i], Ss[i]))
		self.pis = pis
		self.dd = DiscreteDistribution(pis)
	def p(self, x):
		p = 0
		for i in range(self.k):
			p += self.pis[i] * self.gs[i].p(x)
		return p
	def sample(self):
		i = self.dd.sample()
		return self.gs[i].sample()

class EMGauss:
	def __init__(self, x = None, *args, **kwargs):
		if x is not None:
			self.train(x, *args, **kwargs)
	def train(self, x, k, threshold = 1, max = None):
		self.Ns = False
		self.init(x, k, threshold)
		while not self.convergent() and (max is not None and self.count <= max):
			self.count += 1
			self.E()
			self.M()
			self.heuristic()
	def init(self, x, k, threshold):
		self.count = 0
		self.x = x
		m, n = x.shape
		self.n = n
		self.k = k
		self.threshold = threshold
		z = Z(x)
		x = z.z(x)
		gs = KMeans(x, k).groups(x)
		self.mus = np.matrix(np.zeros((m, k), dtype=float))
		self.SIs = []
		SI = covariance(x).I
		for i in range(k):
			try:
				y = z.inv(gs[i])
				self.mus[:,i] = mean(y)
				self.SIs.append(covariance(y).I)
			except KeyError:
				self.mus[:,i] = self.x[:,random.randrange(self.n)]
				self.SIs.append(random.random() * covariance(x).I)
			except np.linalg.LinAlgError:
				self.SIs.append(random.random() * covariance(x).I)
		self.pis = np.ones((1, k), dtype=float) / k
		self.isFirst = True
		self.l = None
	def E(self):
		p = np.zeros((self.n, self.k), dtype=float)
		for i in range(self.k):
			g = Gaussian(self.mus[:,i], self.SIs[i].I)
			for j in range(self.n):
				p[j, i] = g.p(self.x[:,j])
		p *= self.pis
		p = np.matrix(p)
		q = np.array(p * np.matrix(np.ones((self.k, 1), dtype=float)))
		self.gamma = p / q
	def M(self):
		self.Ns = np.ones((1, self.n), dtype=float) * self.gamma
		NI = 1. / np.array(self.Ns)
		self.mus = np.matrix(np.array(self.x * self.gamma) * NI)
		for i in range(self.k):
			y = self.x - self.mus[:,i]
			z = np.matrix(np.array(self.gamma[:,i].T) * np.array(y))
			S = NI[0,i] * (z * y.T)
			try:
				self.SIs[i] = S.I
			except np.linalg.LinAlgError:
				print('LinAlgError')
		self.pis = self.Ns / self.Ns.sum()
	def likelihood(self):
		l = 0
		g = []
		for i in range(self.k):
			g.append(Gaussian(self.mus[:,i], self.SIs[i].I))
		for i in range(self.n):
			_l = 0
			for j in range(self.k):
				_l += self.pis[0, j] * g[j].p(self.x[:,i])
			l += math.log(_l)
		return l
	def convergent(self):
		last = self.l
		self.l = self.likelihood()
		print('likelihood:' + str(self.l))
		print('Ns:' + str(self.Ns))
		print('pis:' + str(self.pis))
		print('vars:' + str([SI.I.diagonal().sum() for SI in self.SIs]))
		if self.isFirst:
			self.isFirst = False
			return False
		if abs(self.l - last) < self.threshold:
			return True
		else:
			return False
	def heuristic(self):
		if (self.Ns < 2).sum() > 0:
			print('abnormal')
			abnormal = self.Ns < 2
			for i in range(self.k):
				if abnormal[0,i] == True:
					print('@' +str(i))
					self.mus[:,i] = self.x[:,random.randrange(self.n)]
					self.SIs[i] = random.random() * covariance(self.x).I
	def get(self):
		mus = [self.mus[:,i] for i in range(self.k)]
		SIs = [self.SIs[i].I for i in range(self.k)]
		pis = [self.pis[0,i] for i in range(self.k)]
		return GaussianMixture(mus, SIs, pis)

class GaussianMixtureEMD(Discriminator):
	def train(self, x0, x1, k, max):
		m0, n0 = x0.shape
		m1, n1 = x1.shape
		x0 = np.vstack((np.zeros(n0), x0))
		x1 = np.vstack((np.ones(n1), x1))
		x = np.hstack((x0, x1))
		em = EMGauss(x, k, max = max)
		self.gm = em.get()
	def do(self, x):
		p0 = self.gm.p(np.vstack((np.array(0), x)))
		p1 = self.gm.p(np.vstack((np.array(1), x)))
		return 0 if p0 > p1 else 1

def emCrossValidation(x0, x1, k, max = 10, bin = 1):
	success = 0
	failure = 0
	r, c0 = x0.shape
	r, c1 = x1.shape
	l = [0] * c0 + [1] * c1
	random.shuffle(l)
	l0 = 0
	l1 = 0
	for i in range(0, c0 + c1, bin):
		r0 = l0 + l[i:i+bin].count(0)
		r1 = l1 + l[i:i+bin].count(1)
		_x0 = np.hstack((x0[:,:l0], x0[:,r0:]))
		_x1 = np.hstack((x1[:,:l1], x1[:,r1:]))
		try:
			d = GaussianMixtureEMD()
			d.train(_x0, _x1, k, max = max)
			for j in range(l0, r0):
				if d.do(x0[:,j]) == 0:
					success += 1
				else:
					failure += 1
			for j in range(l1, r1):
				if d.do(x1[:,j]) == 1:
					success += 1
				else:
					failure += 1
		except Exception as e:
			print(e)
		l0 = r0
		l1 = r1
	return success, failure 

