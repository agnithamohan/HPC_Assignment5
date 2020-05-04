# HPC_Assignment5
## Steps to run : 
- module load openmpi/gnu/4.0.2
- make 
- mpirun --mca btl '^openib' --np <number-of-processes> jac <N> <max-iterations> 
- mpirun --mca btl '^openib' --np <number-of-processes> ssort <N>