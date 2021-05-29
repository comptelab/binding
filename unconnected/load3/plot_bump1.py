#!/usr/bin/python2.7
from bump1_model import *

m=Model()
for models in sys.argv[1:]:
	 m.plot(fd=models)

