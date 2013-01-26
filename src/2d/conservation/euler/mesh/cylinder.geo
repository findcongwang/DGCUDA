Point(1) = {.5,.5,0,0.};
Point(2) = {.5,.4,0,0.};
Point(3) = {.6,.5,0,0.};
Point(4) = {.5,.6,0,0.};
Point(5) = {.4,.5,0,0.};
Point(6) = {0,0,0,0.};
Point(7) = {1,0,0,0.};
Point(8) = {1,1,0,0.};
Point(9) = {0,1,0,0.};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};

Line(5) = {6,7};
Line(6) = {7,8};
Line(7) = {8,9};
Line(8) = {9,6};

Line Loop(9) = {5,6,7,8};
Line Loop(10) = {1,2,3,4};

Plane Surface(11) = {9,10};

Physical Line(10000) = {1,2,3,4};
Physical Line(20000) = {5,6,7,8};

Physical Surface (100) = {11};
