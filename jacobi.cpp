/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string.h>



/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
    int i , j; 
    double tmp, gres = 0.0, lres = 0.0;

    for(int i = 1 ; i <= lN ; i++)
        for(int j = 1; j <= lN ; j++){
            tmp  = 4.0 *  lu[i*lN + j]; 
            tmp -= lu[(i-1)*lN + j]; 
            tmp -= lu[i*lN + (j-1)]; 
            tmp -= lu[(i+1)*lN + j]; 
            tmp -= lu[i*lN + (j+1)];
            tmp *= invhsq; 
            lres += pow((tmp-1.0),2);
        }  

    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}


bool isPowerOf(int n , int val); 


int main(int argc, char * argv[]) {
  int mpirank, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  //printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
# pragma omp parallel
  {
#ifdef _OPENMP
    int my_threadnum = omp_get_thread_num();
    int numthreads = omp_get_num_threads();
#else
    int my_threadnum = 0;
    int numthreads = 1;
#endif
    //printf("Hello, I'm thread %d out of %d on mpirank %d\n", my_threadnum, numthreads, mpirank);
  }

  if(!isPowerOf(p,4) && mpirank == 0){
    printf("Exiting. p must be a power of 4\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  /* compute number of unknowns handled by each process */
  lN = N / sqrt(p);

  if ((N % (int)sqrt(p) != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of square root of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;
/*---------------------------------------------*/
  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

#pragma omp parallel for default(none) shared(lN,lunew,lu,hsq)
    /* Jacobi step for local points */
    for (int i = 1; i <= lN; i++){
        for(int j = 1; j <= lN ; j++){
            lunew[i*lN + j]  = 0.25 * (hsq + lu[(i-1)*lN + j] + lu[i*lN + (j-1)] + lu[(i+1)*lN + j] + lu[i*lN + (j+1)]); 
        }
    }

  if(mpirank < p - sqrt(p)){
      //printf("\n %d sending top to %d\n", mpirank, mpirank + (int)sqrt(p) );
      MPI_Send(&(lunew[1*lN + 1]), lN , MPI_DOUBLE , mpirank + (int)sqrt(p) , 124,  MPI_COMM_WORLD); 
      MPI_Recv(&(lunew[0*lN + 1]), lN , MPI_DOUBLE , mpirank + (int)sqrt(p) , 123,  MPI_COMM_WORLD,&status); 
  } 

  if(mpirank >= sqrt(p)){
      //printf("\n %d sending bottom to %d\n", mpirank, mpirank - (int)sqrt(p) );
      MPI_Send(&(lunew[lN*lN + 1]), lN , MPI_DOUBLE , mpirank - (int)sqrt(p) , 123,  MPI_COMM_WORLD); 
      MPI_Recv(&(lunew[(lN+1)*lN + 1]), lN , MPI_DOUBLE , mpirank - (int)sqrt(p) , 124,  MPI_COMM_WORLD,&status1); 
  }  

  if(mpirank % (int)sqrt(p) != sqrt(p)-1){
      double *temp = (double*)calloc(sizeof(double),lN);
      for( int i = 0 ; i < lN ; i++){
          temp[i] = lunew[(i+1)*lN+lN];
      }
      MPI_Send(temp, lN , MPI_DOUBLE , mpirank + 1 , 123,  MPI_COMM_WORLD);
      double *temp2 = (double*)calloc(sizeof(double),lN);
      MPI_Recv(temp2, lN , MPI_DOUBLE , mpirank + 1 , 124,  MPI_COMM_WORLD,&status);
      for( int i = 1 ; i <= lN ; i++)
        lunew[i*lN+(lN+1)] = temp[i-1];
  }

  if(mpirank % (int)sqrt(p) != 0){
      double *temp = (double*)calloc(sizeof(double),lN);
      for( int i = 0 ; i < lN ; i++){
          temp[i] = lunew[(i+1)*lN+1];
      }
      MPI_Send(temp, lN , MPI_DOUBLE , mpirank - 1 , 124,  MPI_COMM_WORLD);
      double *temp2 = (double*)calloc(sizeof(double),lN);
      MPI_Recv(temp2, lN , MPI_DOUBLE , mpirank - 1 , 123,  MPI_COMM_WORLD,&status1);
      for( int i = 1 ; i <= lN ; i++)
        lunew[i*lN+0] = temp[i-1];
  }
  
 
    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
   }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}


bool isPowerOf(int n, int val ) 
{ 
    if(n == 0) 
        return 0; 
    while(n != 1) 
    {  
        if(n % val != 0) 
            return 0; 
        n = n / val;  
    } 
    return 1; 
} 