import time
class mytimer(object):
	def __init__(self):
		self.lastTime = int(time.time())
		self.buffer = 15
	
	def isTransition(self):
		if int(time.time()) - self.buffer > self.lastTime:
			self.lastTime = int(time.time())
			return True
		return False

	def setBuffer(self,seconds):
		self.buffer = seconds