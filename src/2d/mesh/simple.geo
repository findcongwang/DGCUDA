// Gmsh project created on Fri Mar  9 12:34:28 2012
l = .01;
Point(1) = {0, 0, 0, l};
Point(2) = {0, 1, 0, l};
Point(3) = {1, 1, 0, l};
Point(4) = {1, 0, 0, l};
Line(1) = {3, 4};
Line(2) = {4, 1};
Line(3) = {1, 2};
Line(4) = {2, 3};
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};
