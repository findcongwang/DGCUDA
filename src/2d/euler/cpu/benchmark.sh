#!/bin/bash
time ./cpueuler -T 10 -n 0 mesh/sv1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 1 mesh/sv1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 2 mesh/sv1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 3 mesh/sv1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 4 mesh/sv1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 5 mesh/sv1.pmsh output/uniform.out 

time ./cpueuler -T 10 -n 0 mesh/sv1refined.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 1 mesh/sv1refined.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 2 mesh/sv1refined.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 3 mesh/sv1refined.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 4 mesh/sv1refined.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 5 mesh/sv1refined.pmsh output/uniform.out 

time ./cpueuler -T 10 -n 0 mesh/sv1refined1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 1 mesh/sv1refined1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 2 mesh/sv1refined1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 3 mesh/sv1refined1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 4 mesh/sv1refined1.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 5 mesh/sv1refined1.pmsh output/uniform.out 

time ./cpueuler -T 10 -n 0 mesh/sv1refined2.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 1 mesh/sv1refined2.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 2 mesh/sv1refined2.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 3 mesh/sv1refined2.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 4 mesh/sv1refined2.pmsh output/uniform.out 
time ./cpueuler -T 10 -n 5 mesh/sv1refined2.pmsh output/uniform.out 
