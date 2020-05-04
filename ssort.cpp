// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
using namespace std; 

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  /*-------*/
  //int N = 100;
    int N=100; 
  sscanf(argv[1], "%d", &N);


  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));


  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  double tt = MPI_Wtime();
  // sort locally
  std::sort(vec, vec+N);

  int *sample = (int*)malloc(N*sizeof(int)); 


  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int skip = N/p; 
  for(int i = skip,j=0 ; i< N ; i+=skip,j++){
    sample[j] = vec[i]; 
  }

  int * sample_recv; 
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  if(rank==0)
    sample_recv = (int*)malloc(p*(p-1)*sizeof(int)); 
  
  MPI_Gather(sample,p-1,MPI_INT,sample_recv,p-1,MPI_INT,0,MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int *splitters = (int*)malloc((p-1)*sizeof(int));
  if(rank==0){
    std::sort(sample_recv, sample_recv+p*(p-1));
    int skip = p*(p-1)/p;
    
    for(int i = skip,j=0 ; i < p*(p-1) ; i+=skip,j++){
      splitters[j] = sample_recv[i]; 
    }
  }

  // root process broadcasts splitters to all other processes
   MPI_Bcast(splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  int **indices = (int**)malloc((p)*sizeof(int*)); 
  indices[0] = vec ; 
  for(int i=0; i<p-1 ;i++){
    indices[i+1] = std::lower_bound(vec, vec+N , splitters[i]); 
  }
 
  int *count = (int*)malloc(p*sizeof(int)); 

  for(int i = 0; i < p-1 ; i++)
  {
    count[i] = indices[i+1] - indices[i]; 
  }
  count[p-1] = (vec+N) - indices[p-1]; 

  int *disp = (int*)malloc(p*sizeof(int)); 
  disp[0] = vec-vec; 
  for(int i = 1;  i < p ; i++){
    disp[i] = indices[i] - vec; 
  }



  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int *recv_count = (int*)malloc(p*sizeof(int)); 

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data



   MPI_Alltoall(count,  1 , MPI_INT, recv_count, 1, MPI_INT, MPI_COMM_WORLD); 
  int n = 0; 
  for(int i = 0 ; i < p ; i++)
    n += recv_count[i]; 
  int *bucket_vals = (int*)malloc(n*sizeof(int)); 

  int* rdisp = (int*)calloc(p,sizeof(int)); 
  for(int i = 1 ; i < p ; i++)
    rdisp[i] = rdisp[i-1] + recv_count[i-1]; 
  
  MPI_Alltoallv(vec, count, disp , MPI_INT , bucket_vals , recv_count , rdisp , MPI_INT, MPI_COMM_WORLD); 


  // do a local sort of the received data
  std::sort(bucket_vals, bucket_vals+n); 
  MPI_Barrier(MPI_COMM_WORLD); 
  double elapsed = MPI_Wtime() - tt;
  if(rank == 0){
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  // every process writes its result to a file
  {
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "SortedValues/output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }
    for(int i = 0; i < n; ++i)
      fprintf(fd, "  %d\n", bucket_vals[i]);

    fclose(fd);
  }

  free(vec);
  MPI_Finalize();
  return 0;
}
