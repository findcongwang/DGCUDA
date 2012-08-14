#!/usr/bin/python
"""
genmesh.py

This file is responsible for reading the contents of the .msh file 
and translating them into something usable for CUDA.
"""

from sys import argv

def genmesh(inFilename, outFilename):
    inFile  = open(inFilename, "rb")
    outFile = open(outFilename, "wb")

    line = inFile.readline()
    while line != "$Nodes\n":
        line = inFile.readline()

    # the next line is the number of vertices
    num_verticies = int(inFile.readline())

    vertex_list = []
    for i in xrange(0,num_verticies):
        s = inFile.readline().split()
        vertex_list.append((float(s[1]), float(s[2])))
    
    # next two lines are just filler
    inFile.readline()
    inFile.readline()

    # next line is the number of elements
    num_elements = int(inFile.readline())

    edge_list = []
    elem_list = []
    boundary_list = []
    # add the vertices for each element into elem_list
    for i in xrange(0,num_elements):
        s = inFile.readline().split()
        if len(s) == 7:
            boundary = int(s[3])
            v1 = int(s[5]) - 1
            v2 = int(s[6]) - 1
            edge_list.append((vertex_list[v1], vertex_list[v2]))
            boundary_list.append(boundary)

        if len(s) == 8:
            v1 = int(s[5]) - 1
            v2 = int(s[6]) - 1
            v3 = int(s[7]) - 1
            elem_list.append((vertex_list[v1], vertex_list[v2], vertex_list[v3]))

    # write the number of elements to the file
    outFile.write(str(len(elem_list)) + "\n")

    # write the elements in elem_list to outFile
    for elem in elem_list:
        # check to see if any edge is a boundary edge the boundary edges in edge_list to outFile
        side_number  = -1
        boundary_idx = 0
        for edge, boundary in zip(edge_list, boundary_list):
            if ((edge[0][0] == elem[0][0] and edge[0][1] == elem[0][1]) or (edge[0][0] == elem[0][1] and edge[0][1] == elem[0][0])):
                side_number = 0
                boundary_idx = boundary
                break
            elif ((edge[0][0] == elem[1][0] and edge[0][1] == elem[1][1]) or (edge[0][0] == elem[1][1] and edge[0][1] == elem[1][0])):
                side_number = 1
                boundary_idx = boundary
                break
            elif ((edge[0][0] == elem[2][0] and edge[0][1] == elem[2][1]) or (edge[0][0] == elem[2][1] and edge[0][1] == elem[2][0])):
                side_number = 2
                boundary_idx = boundary
                break

        outFile.write("%f %f %f %f %f %f %i %i\n" % (elem[0][0], elem[0][1],
                                               elem[1][0], elem[1][1],
                                               elem[2][0], elem[2][1],
                                               side_number, boundary_idx))

    outFile.close()


if __name__ == "__main__":
    try:
        inFilename  = argv[1] 
        outFilename = argv[2]
        genmesh(inFilename, outFilename)
    except:
        print "usage: genmesh.py [infile] [outfile]"
