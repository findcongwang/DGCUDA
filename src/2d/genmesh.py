"""
genmesh.py

This file is responsible for reading the contents of the .msh file 
and translating them into something usable for CUDA.
"""

from sys import argv

if __name__ == "__main__":
    try:
        inFilename  = argv[1] 
        outFilename = argv[2]
        genmesh(inFilename, outFilename)
    except:
        print "usage: genmesh.py [infile] [outfile]"
