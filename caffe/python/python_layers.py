import caffe
import numpy as np

class Layer(caffe.Layer):
	def setup(self, bottom, top):
		if self.param_str:
			params = eval(self.param_str)
			if isinstance(params, dict):
				for p,v in params.items():
					setattr(self, p, v)

class Meshgrid(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str)
		self.width = params['width']
		self.height = params['height']
		self.batch = params['batch']
	def reshape(self, bottom, top):
		assert len(bottom) == 0, "No bottom accepted"
		assert len(top) == 1, "Only one top accepted"
		top[0].reshape(self.batch, 2, self.height, self.width)
		# top[1].reshape(self.batch, 1, self.height, self.width)
	def forward(self, bottom, top):
		gx, gy = np.meshgrid(range(self.width), range(self.height))
		gxy = np.concatenate((gx[None,:,:], gy[None,:,:]), axis=0)
		top[0].data[...] = gxy[None, :, :, :]
		# top[1].data[...] = gy[None,None,:,:]
	def backward(self, top, propagate_down, bottom):
		pass

class MeanVals(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str)
		self.width = params['width']
		self.height = params['height']
		self.batch = params['batch']
	def reshape(self, bottom, top):
		assert len(bottom) == 0, "No bottom accepted"
		assert len(top) == 1, "Only one top accepted"
		top[0].reshape(self.batch, 3, self.height, self.width)
	def forward(self, bottom, top):
		m1 = 104 * np.ones((self.height, self.width))
		m2 = 117 * np.ones((self.height, self.width))
		m3 = 123 * np.ones((self.height, self.width))
		mall = np.concatenate((m1[None,:,:], m2[None,:,:], m3[None,:,:]), axis=0)
		top[0].data[...] = mall[None, :, :, :]
		# top[1].data[...] = gy[None,None,:,:]
	def backward(self, top, propagate_down, bottom):
		pass

class MaskedL1Loss(MaskedL2Loss):
	def forward(self, bottom, top):
		A = bottom[0].data
		B = bottom[1].data
		N = A.shape[0]
		M = np.ones(N)
		if len(bottom)>2:
			M = (bottom[2].data > 0.5).astype(np.float32)
		top[0].data[0] = 0
		N = A.shape[0]
		for n in range(N):
			a, b, m = 1*A[n], B[n], M[n]
			# if self.do_shift:
			# 	shift = (np.median(b-a))*0.99
			# 	a += shift
			# scale = 1
			# if self.do_scale:
			# 	# It should be the weighted median of b / a weighted by |a|
			# 	# This is a very hacky approximation ...
			# 	for it in range(10):
			# 		w = m/(abs(a-b)+1e-10)
			# 		scale = np.abs(np.mean(w*a*b) / (np.mean(w*a*a)+1e-10))*0.99 + 0.01
			# 		a *= scale
			# 	#assert False, "Scaling not implemented"
			norm = 1.0 / (np.sum(m+a*0)+1e-10) / N
			top[0].data[0] += np.sum(m*abs(a-b)) * norm
			diff = m*np.sign(a-b) * norm
			# bottom[0].diff[n] = scale * diff
			bottom[0].diff[n] = diff
			bottom[1].diff[n] = -diff

class ShapenetSamplerNoEle(caffe.Layer):
	def setup(self, bottom, top):
		self.azi_rel = range(-180, 181, 20)
		self.azi_abs = range(0, 360, 5)
		self.ele_abs = range(0, 31, 5)

		params = eval(self.param_str)
		self.batch = params['batch']
		self.img_size = params['img_size']
		self.num_tform = params['num_tform']
		assert self.num_tform == len(self.azi_rel), "num_tform not matching"
		self.shapeList = pickle.load(open(params['shapeListFile'], 'rb'))
	def reshape(self, bottom, top):
		assert len(bottom) == 0, "No bottom supported"
		assert len(top) == 5, "Only five tops supported"
		top[0].reshape(self.batch, 3, self.img_size, self.img_size)
		top[1].reshape(self.batch, 3, self.img_size, self.img_size)
		top[2].reshape(self.batch, 1, self.img_size, self.img_size)
		top[3].reshape(self.batch, 1, self.img_size, self.img_size)
		top[4].reshape(self.batch, self.num_tform)
	def forward(self, bottom, top):
		def clip_angles(ang):
		    if ang >= 360:
		        ang = ang - 360
		    if ang < 0:
		        ang = ang + 360
		    return ang
		nsample = 0
		b_srcImg = np.zeros((self.batch, 3, self.img_size, self.img_size))
		b_tgtImg = np.zeros((self.batch, 3, self.img_size, self.img_size))
		b_tform = np.zeros((self.batch, self.num_tform))
		b_srcMask = np.zeros((self.batch, 1, self.img_size, self.img_size))
		b_tgtMask = np.zeros((self.batch, 1, self.img_size, self.img_size))
		while nsample < self.batch:
			srcAzi = self.azi_abs[np.random.randint(len(self.azi_abs))]
			srcEle = self.ele_abs[np.random.randint(len(self.ele_abs))]
			aziRelID = np.random.randint(len(self.azi_rel))
			aziVec = np.zeros((len(self.azi_rel)))
			aziVec[aziRelID] = 1
			b_tform[nsample] = aziVec
			tgtAzi = clip_angles(srcAzi + self.azi_rel[aziRelID])
			tgtEle = srcEle
			shapeID = np.random.randint(len(self.shapeList))
			srcInput = skio.imread(self.shapeList[shapeID] + '/model_views/' + str(srcAzi) + '_' + str(srcEle) + '.png').astype(np.float32)
			b_srcImg[nsample] = srcInput[:,:,:3].transpose((2, 0, 1))
			b_srcMask[nsample] = (srcInput[:,:,3]>0).reshape((1, self.img_size, self.img_size))
			tgtInput = skio.imread(self.shapeList[shapeID] + '/model_views/' + str(tgtAzi) + '_' + str(tgtEle) + '.png').astype(np.float32)
			b_tgtImg[nsample] = tgtInput[:,:,:3].transpose((2, 0, 1))
			b_tgtMask[nsample] = (tgtInput[:,:,3]>0).reshape((1, self.img_size, self.img_size))
			nsample += 1
		b_srcImg = b_srcImg + 255 * (1 - b_srcMask)
		b_tgtImg = b_tgtImg + 255 * (1 - b_tgtMask)
		# if True:
		# 	import scipy.io
		# 	i = np.random.randint(10)
		# 	sim = b_srcImg[i].transpose((1,2,0))
		# 	tim = b_tgtImg[i].transpose((1,2,0))
		# 	smask = b_srcMask[i]
		# 	tmask = b_tgtMask[i]
		# 	scipy.io.savemat('/tmp/debug/' + str(i) + '.mat', {'sim':sim, 'tim':tim, 'smask':smask, 'tmask':tmask})
		top[0].data[...] = b_srcImg
		top[1].data[...] = b_tgtImg
		top[2].data[...] = b_srcMask
		top[3].data[...] = b_tgtMask
		top[4].data[...] = b_tform
	def backward(self, top, propagate_down, bottom):
		pass
