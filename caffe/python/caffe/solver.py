from __future__ import print_function
import caffe

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

class Solver:
	def __init__(self, prototxt, final_file=None, snap_file=None, solver='Adam', log_file=None, **kwargs):
		self.running = False
		self.force_snapshot = False
		self.last_interrupt = 0
		self.final_file, self.snap_file = final_file, snap_file
		if final_file is None:
			print("Are you sure you dont want to save the model?")
		solver_str = 'train_net: "%s"\n'%prototxt
		if solver is not None:
			# Get the cases right
			if solver.upper() == "ADAM": solver = "Adam"
			if solver.upper() == "ADADELTA": solver = "AdaDelta"
			if solver.upper() == "ADAGRAD": solver = "AdaGrad"
			if solver.upper() == "NESTEROV": solver = "Nesterov"
			if solver.upper() == "RMSPROP": solver = "RMSProp"
			if solver.upper() == "SGD": solver = "SGD"
			solver_str += 'type: "%s"\n'%solver
			if solver == "RMSProp":
				if 'rms_decay' not in kwargs:
					kwargs['rms_decay'] = 0.9
			if solver == "SGD":
				if 'momentum' not in kwargs:
					kwargs['momentum'] = 0.9
			if solver == "Adam":
				if 'momentum' not in kwargs:
					kwargs['momentum'] = 0.9
				if 'momentum2' not in kwargs:
					kwargs['momentum2'] = 0.99
		if 'base_lr'   not in kwargs: kwargs['base_lr'] = 0.001
		if 'lr_policy' not in kwargs: kwargs['lr_policy'] = 'fixed'
		for i in kwargs:
			if isinstance(kwargs[i], str):
				solver_str += '%s: "%s"\n'%(i, kwargs[i])
			elif isinstance(kwargs[i], int):
				solver_str += '%s: %d\n'%(i, kwargs[i])
			else:
				solver_str += '%s: %f\n'%(i, kwargs[i])
		self.solver = caffe.get_solver_from_string(solver_str)
		self.solver.add_callback(self.on_start, self.on_gradient)

		self.log = None
		if log_file:
			self.log = open(log_file, 'w')

	def on_start(self):
		pass

	def on_gradient(self):
		pass

	def on_display(self):
		pass

	def after_step(self):
		pass

	def stop(self):
		print("Solver shutting down ...")
		self.running = False

	def try_stop(self):
		from time import time
		if time() - self.last_interrupt < 5:
			return self.stop()
		print("Snapshoting (To exit interrupt twice within 5 sec)")
		self.force_snapshot = True
		self.last_interrupt = time()

	def run(self, nit, show_debug=False, print_interval=1, snap_interval=60):
		import signal
		from time import time
		import numpy as np
		self.running = True
		avg_weight = 0.95

		# Register interrupt
		signal.signal(signal.SIGINT, lambda *a:self.try_stop())

		# Train
		s = self.solver
		s.net.save(self.snap_file)
		loss_blobs = [b for b in s.net.blob_loss_weights
		              if s.net.blob_loss_weights[b]!=0]
		loss_weight = np.array([s.net.blob_loss_weights[b] for b in loss_blobs])
		t0, t1, last_it = 0, 0, 0
		n = np.zeros(len(loss_blobs))
		sl = np.zeros(len(loss_blobs))
		for it in range(nit):
			s.step(1)
			self.after_step()
			l = np.array([np.sum(s.net.blobs[b].data) for b in loss_blobs])
			n = avg_weight*n+1
			sl = avg_weight*sl+l
			ll = sl / n
			if time()-t0 > print_interval:
				print('[%s% 5d%s it  |  %s% 3d%s it / sec]\tl: %s%10.5g%s \tal: %s%10.5g%s  [%s]'%(bcolors.BOLD, it, bcolors.ENDC, bcolors.BOLD, (it-last_it)/print_interval, bcolors.ENDC, bcolors.OKGREEN, l.dot(loss_weight), bcolors.ENDC, bcolors.OKGREEN, ll.dot(loss_weight), bcolors.ENDC, ' '.join(['%s = %7.2g'%(b,v) for b,v in zip(loss_blobs,ll)])))
				if self.log is not None:
					print('[% 5d it  |  % 3d it / sec]\tl: %10.5g  [%s]'%(it, (it-last_it)/print_interval, l.dot(loss_weight), ' '.join(['%s = %7.2g'%(b,v) for b,v in zip(loss_blobs,l)])),
				      ' \tal: %10.5g  [%s]'%(ll.dot(loss_weight), ' '.join(['%s = %7.2g'%(b,v) for b,v in zip(loss_blobs,ll)])),file=self.log)
					self.log.flush()
				if show_debug:
					print(' '*10+'\t','  '.join(['% 5s'%b[:5] for b in s.net.blobs]))
					print(' '*6+'data\t', '  '.join(['%5.1g'%np.sum(np.abs(s.net.blobs[b].data)) for b in s.net.blobs]))
					print(' '*6+'diff\t', '  '.join(['%5.1g'%np.sum(np.abs(s.net.blobs[b].diff)) for b in s.net.blobs]))
					lrs = list(zip(s.net.layers, s.net._layer_names))
					print(' '*10+'\t','  '.join(['% 11s'%n[:5] for l,n in lrs if len(l.blobs)>0]))
					print(' '*6+'data\t', '  '.join(['%5.1g/%5.1g'%(np.sum(np.abs(l.blobs[0].data)),np.sum(np.abs(l.blobs[-1].data))) for l,n in lrs if len(l.blobs)>0]))
					print(' '*6+'diff\t', '  '.join(['%5.1g/%5.1g'%(np.sum(np.abs(l.blobs[0].diff)),np.sum(np.abs(l.blobs[-1].diff))) for l,n in lrs if len(l.blobs)>0]))
					print()
				self.on_display()
				import sys
				sys.stdout.flush()
				last_it = it
				t0 = time()
			if self.snap_file is not None and (time()-t1 > snap_interval or self.force_snapshot):
				# Snapshot
				s.net.save(self.snap_file)
				self.force_snapshot = False
				t1 = time()
			if not self.running:
				break
		if self.final_file is not None:
			s.net.save(self.final_file)
		signal.signal(signal.SIGINT, signal.SIG_DFL)
		return s.net
