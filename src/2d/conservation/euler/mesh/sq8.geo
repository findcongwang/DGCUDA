lc =0.1;
Point(1) = {0,1.,0,lc};
Point(2) = {0,0,0,lc};
Point(3) = {1,0,0,lc};
Point(4) = {1,1,0,lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(8) = {2,3,4,1};
Plane Surface(9) = {8};

Transfinite Line {2,3,4,1} = 9; //number of points on lines 1,2,3,4; it overrides spacing given by lc;

Transfinite Surface {9} = {2,3,4,1};

/*if the next line is uncommennented, gmsh will produce quads.*/
///Recombine Surface {9};

Physical Surface (100) = {9};
Physical Line (11) = {1};
Physical Line (55) = {2};
Physical Line (33) = {3};
Physical Line (77) = {4};





