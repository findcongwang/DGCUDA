#!/usr/bin/python2
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
    while "$Nodes" not in line:
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

        # these are sides
        if len(s) == 7:
            boundary = int(s[3])
            v1 = int(s[5]) - 1
            v2 = int(s[6]) - 1
            edge_list.append((vertex_list[v1], vertex_list[v2]))
            boundary_list.append(boundary)

        # and these are elements
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

            x1 = edge[0][0]
            y1 = edge[0][1]

            x2 = edge[1][0]
            y2 = edge[1][1]

            elem_x1 = elem[0][0]
            elem_y1 = elem[0][1]

            elem_x2 = elem[1][0]
            elem_y2 = elem[1][1]

            elem_x3 = elem[2][0]
            elem_y3 = elem[2][1]

            if ((x1 == elem_x1 and y1 == elem_y1 and x2 == elem_x2 and y2 == elem_y2) or 
               ( x1 == elem_x2 and y1 == elem_y2 and x2 == elem_x1 and y2 == elem_y1)):
                side_number = 0
                boundary_idx = boundary
            elif ((x1 == elem_x2 and y1 == elem_y2 and x2 == elem_x3 and y2 == elem_y3) or 
                 ( x1 == elem_x3 and y1 == elem_y3 and x2 == elem_x2 and y2 == elem_y2)):
                side_number = 1
                boundary_idx = boundary
            elif ((x1 == elem_x1 and y1 == elem_y1 and x2 == elem_x3 and y2 == elem_y3) or 
                 ( x1 == elem_x3 and y1 == elem_y3 and x2 == elem_x1 and y2 == elem_y1)):
                side_number = 2
                boundary_idx = boundary

        outFile.write("%f %f %f %f %f %f %i %i\n" % (elem[0][0], elem[0][1],
                                               elem[1][0], elem[1][1],
                                               elem[2][0], elem[2][1],
                                               side_number, boundary_idx))

    outFile.close()


if __name__ == "__main__":
    inFilename  = argv[1] 
    outFilename = argv[2]
    genmesh(inFilename, outFilename)
