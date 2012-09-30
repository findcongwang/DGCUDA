"""
edge.py
    The data structure for the edges.
"""

class Edge:
    """
    Variables:
        idx - unique idx to identify the edge
        leftCell - the idx of the cell on the edge's left side
        rightCell - the idx of the cell on the edge's right side
        isBoundary - boolean indicating a boundary cell
        isPeriodic - if the boundary cell is periodic
    """

    def __init__(self, idx, leftCell, rightCell, isBoundary = False, isPeriodic = False):
        self.idx        = idx
        self.leftCell   = leftCell
        self.rightCell  = rightCell
        self.isBoundary = isBoundary
        self.isPeriodic = isPeriodic
        init

    def initIntegrationPoints(self):
