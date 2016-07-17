* Purpose: Use iterative depth-first search and MPI to solve an
            instance of the travelling salesman problem.  This version 
            partitions the search tree using breadth-first search.
            Then each process searches its assigned subtree.  There
            is no reassignment of tree nodes.  This version also attempts
            to reuse deallocated tours.  The best tour structure
            is broadcast using a loop of MPI_Bsends.
 
* Compile:  mpicc -g -Wall -o mpi_tsp main.cpp
* Usage:    mpiexec -n <proc count> mpi_tsp <matrix_file>
 
* Input:    From a user-specified file, the number of cities
            followed by the costs of travelling between the
            cities organized as a matrix:  the cost of
            travelling from city i to city j is the ij entry.
            Costs are nonnegative ints.  Diagonal entries are 0.
* Output:   The best tour found by the program and the cost
            of the tour.
