"""
readmesh.py
    Reads the mesh files and initializes the data structures for holding these.
"""

def readmesh(filename):
    """ 
    Takes the mesh and returns the data structures.
    """
    f = open(filename, 'rb')
    cells = []
    edges = []

    # create each cell and edge, etc
    for line in f:
        
    return cells, edges


