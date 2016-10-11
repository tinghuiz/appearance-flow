from __future__ import print_function
from time import time

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def time_net(net, NIT=100, top=5):
	import numpy as np
	import caffe
	if hasattr(caffe, 'wait_for_cuda'):
		wait_for_cuda = caffe.wait_for_cuda
	else:
		print( 'wait_for_cuda function recommended for accurate timing (https://github.com/philkr/caffe/tree/wait_for_cuda)' )
		def wait_for_cuda(): pass
	L = net._layer_names
	fwd = {n:[] for n in L}
	bck = {n:[] for n in L}
	T0 = time()
	for i in range(NIT):
		for j,n in enumerate(L):
			t0 = time()
			net._forward(j,j)
			wait_for_cuda()
			fwd[n].append(time()-t0)
		for j,n in list(enumerate(L))[::-1]:
			t0 = time()
			net._backward(j,j)
			wait_for_cuda()
			bck[n].append(time()-t0)
	top = min(top, len(fwd)+len(bck))
	T = sorted([np.mean(v) for v in fwd.values()] + [np.mean(v) for v in bck.values()])[-top]
	T0 = time()-T0
	print("%s%0.1f%s it / sec    [%s%0.1f%s ms / it]"%(bcolors.BOLD+bcolors().FAIL, NIT / T0, bcolors.ENDC, bcolors.BOLD+bcolors().FAIL, 1000*T0 / NIT, bcolors.ENDC))
	for n in L:
		cf, cb = bcolors.OKGREEN, bcolors.OKGREEN
		if np.mean(fwd[n]) >= T: cf = bcolors.BOLD+bcolors().FAIL
		if np.mean(bck[n]) >= T: cb = bcolors.BOLD+bcolors().FAIL
		print('  %30s  \t %s%0.2f \261 %0.1f%s ms  \t %s%0.2f \261 %0.1f%s ms'%(n, cf, 1000*np.mean(fwd[n]), 1000*np.std(fwd[n]), bcolors.ENDC, cb, 1000*np.mean(bck[n]), 1000*np.std(bck[n]), bcolors.ENDC))

if __name__ == "__main__":
	import argparse
	from os import path
	import caffe
	parser = argparse.ArgumentParser(description="Visualize decompositions on sintel")
	parser.add_argument('input_dir', help='input directory')
	parser.add_argument('-n', type=int, default=100, help='Number of iterations')
	parser.add_argument('-t', type=int, default=5, help='Highlight the top t times')
	parser.add_argument('-gpu', type=int, help='What GPU do we test on')
	args = parser.parse_args()

	caffe.set_mode_gpu()
	if args.gpu is not None:
		caffe.set_device(args.gpu)

	if path.isfile(args.input_dir):
		net = caffe.Net(args.input_dir, caffe.TRAIN)
	else:
		net = caffe.Net(args.input_dir+'trainval.prototxt', caffe.TRAIN)
	time_net(net)
