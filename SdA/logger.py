import sys

class Logger(object):
	
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.err = sys.stderr
                sys.stderr = self
		self.log = open(filename, "a", 0)

	def write(self, message):
		self.terminal.write(message)
                self.log.write(message)

	def reboot(self):
                sys.stdout = self.terminal
		sys.stderr = self.err
	        self.log.close()

