import sys
import cPickle
import utils

for line in sys.stdin: 

    line = eval(line)

    print utils.compress(line)


