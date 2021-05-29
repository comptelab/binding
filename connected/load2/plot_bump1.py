#!/usr/bin/python2.7
from bump3_model2 import *

m=Model()
for models in sys.argv[1:]:
	 m.plot(fd=models)

