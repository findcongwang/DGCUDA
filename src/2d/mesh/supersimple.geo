Point(1) = {-0.1, 0.5, 0, 1.0};
Point(2) = {0.2, 0.5, 0, 1.0};
Point(3) = {0.6, 0.5, 0, 1.0};
Point(4) = {0.8, 0.5, 0, 1.0};
Point(5) = {1, 0.5, 0, 1.0};
Point(6) = {1.1, 0.5, 0, 1.0};
Point(7) = {0.4, 0.5, 0, 1.0};
Point(8) = {0.1, 0.5, 0, 1.0};
Point(9) = {1, 0.5, 0.2, 1.0};
Point(10) = {1, 0.5, -0.3, 1.0};
Point(11) = {0.1, 0.5, -0.2, 1.0};
Point(12) = {0.1, 0.5, 0.2, 1.0};
Circle(1) = {12, 1, 11};
Circle(2) = {12, 1, 1};
Point(13) = {1.8, 0.5, 0.1, 1.0};
Point(1) = {1, 1, 0, 1.0};
Point(2) = {0, 0, 0, 1.0};
Point(3) = {1, 0, 0, 1.0};
Point(4) = {1, 0, 0, 1.0};
Line(1) = {1, 3};
Point(5) = {0, 0, 0, 1.0};
Point(6) = {1, 1, 0, 1.0};
Point(7) = {1, 0, 0, 1.0};
Delete {
  Point{2};
}
Delete {
  Point{5};
}
Delete {
  Point{1};
}
Delete {
  Point{1};
}
Delete {
  Point{1};
}
Delete {
  Point{1};
}
Delete {
  Point{1, 3};
}
Delete {
  Point{3};
}
Delete {
  Point{3};
}
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};
Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 1};
Line(4) = {1, 2};
Line Loop(5) = {2, 3, 4, 1};
Plane Surface(6) = {5};
