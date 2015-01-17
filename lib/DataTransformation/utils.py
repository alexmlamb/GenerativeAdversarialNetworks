import cPickle

specialStr = str(hash("a"))

def uncompress(x): 
    return cPickle.loads(x.replace(specialStr, "\n"))

def compress(x): 
    return cPickle.dumps(x, protocol = cPickle.HIGHEST_PROTOCOL).replace("\n", specialStr)
