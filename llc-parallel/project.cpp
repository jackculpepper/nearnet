
#include "util.h"
#include "test.h"

#ifdef PARALLEL
#include <mpi.h>
#endif

#include <flann/flann.hpp>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permute.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <ctime>

typedef gsl_matrix* Matrix;

Matrix All_X; //all the data in the schedule
Matrix X; // data in the batch
Matrix A; // basis functions
Matrix S; // coefficients
//Matrix G; // gradient update
//Matrix G_sum; // for collective operations

size_t L;// = 32*32*3;//size of each image
size_t M;// = 3072; // number of basis components
size_t NUM_IMAGES;

// We'll have a schedule of iterations
// and process B images each time
//size_t Schedule;//=100; 
size_t B;// = 100;//batch size

int K;//=50; // K-nearest neighbour
double lambda=0.01; //regularization parameter for matrix inverse
double beta=0.01;
//double eta = 0.0; // jack case
double eta = 0.005; // learning rte
double target_angle = 0.1;
  
int num_iterations;
int write_every_n_time;

string dataset;
string dataset_list;
string A_initfile;

string run_prefix;
string prefix;

ofstream logfile;
stringstream logstring;
ofstream timefile;
stringstream timestring;

Matrix regularized;
Matrix regularized_inverse;
Matrix temp_LU;
gsl_permutation *p;
Matrix Dtx;
Matrix R;
Matrix T;
Matrix D;
gsl_vector *A_old;
gsl_vector *A_new;  

Matrix G;
Matrix G_sum;

gsl_matrix_view X_view;

int mpi_rank=0;
int mpi_size=1;

bool restart = false;

double coefficient_time, gradient_time, update_time;
double total_iteration_time;
double solve_time, distance_time;
clock_t start0, finish0;
clock_t start1, finish1;
clock_t start2, finish2;
clock_t start3, finish3;
clock_t start4, finish4;
clock_t start5, finish5;

void initialize_computation_vars() {
  //cout<<"Init computation vars"<<endl;
  //  cout<<"K is "<<K<<endl;
  regularized = gsl_matrix_alloc(K,K);
  regularized_inverse = gsl_matrix_alloc(K,K);
  temp_LU = gsl_matrix_alloc(K,K);
  p = gsl_permutation_alloc(K);
  Dtx = gsl_matrix_alloc(K,1);
  T = gsl_matrix_alloc(K,1);
  D = gsl_matrix_alloc(L, K);

}

void free_computation_vars() {
  //cout<<"Free computation vars"<<endl;
  gsl_matrix_free(D);
  gsl_matrix_free(T);
  gsl_matrix_free(Dtx);
  gsl_permutation_free(p);
  gsl_matrix_free(temp_LU);
  gsl_matrix_free(regularized_inverse);
  gsl_matrix_free(regularized);
}


void print_parameters() {

  logstring<<"***********************************************"<<endl;
  logstring<<"***********************************************"<<endl;
  //  logstring<<"Starting execution on "<<B<<" images"<<endl;
  //  logstring<<"Number of schedules is "<<Schedule<<", #iterations is "<<num_iterations<<endl;
  logstring<<"Num Images is "<<NUM_IMAGES<<endl;
  logstring<<"Batch size is "<<B<<endl;
  logstring<<"Each image is sized "<<L<<endl;
  logstring<<"Asked code to produce "<<M<< " basis components"<<endl;
  logstring<<"lambda= "<<lambda<<endl;
  logstring<<"Starting K= "<<K<<endl;
  logstring<<"Starting target_angle="<<target_angle<<endl;
  logstring<<"Starting eta= "<<eta<<endl;
  logstring<<"Write Interval "<<write_every_n_time<<endl;
  logstring<<"***********************************************"<<endl; 
  logstring<<"***********************************************"<<endl;
  
}

void initialize_stuff() {
  srand((mpi_rank+1)*100);

  print_parameters();
  
  logstring<<"Reading X data from file"<<endl;
  All_X = gsl_matrix_alloc(L,NUM_IMAGES);

  // need to determine if we need to read doubles or uint8
  // we'll hack this to look for patterns in the filename for now
  const char* pch=0;
  string type_bruno="bruno";
  pch=strstr(dataset.c_str(),type_bruno.c_str());
  if (pch)
    util::read_double_matrix_from_file(All_X,dataset);
  else
    util::read_uint8_matrix_from_file(All_X,dataset);

  logstring<<"Reading All_X from "<<dataset<<endl;
  util::check_matrix(All_X,"All_X read from file");
  
  util::removeMeanFromImages(All_X);
  util::standardizeImages(All_X);
  util::check_matrix(All_X,"All_X after standardization");
  
  util::shuffleMatrixColumns(All_X);
  
  logstring<<"Initializing A"<<endl;
  A = gsl_matrix_alloc(L,M);
  
  if (mpi_rank==0) {

    if (restart) {
      //util::read_uint8_matrix_from_file(A,A_initfile);
      util::read_double_matrix_from_file(A,A_initfile);
      logstring<<"Setting A from file "<<A_initfile<<endl;
    }
    else {
      util::randomize(A); /* random initialization */
      logstring<<"Setting A to random noise"<<endl;
      util::removeMeanFromImages(A);
      util::standardizeImages(A);
    }

  }
  
#ifdef PARALLEL  
  MPI_Bcast(A->data, A->size1*A->size2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  A_old = gsl_vector_alloc(L*M);
  A_new = gsl_vector_alloc(L*M);

  logstring<<"Initializing Gradient matrices"<<endl;
  G = gsl_matrix_alloc(L,M);
  G_sum =gsl_matrix_alloc(L,M);
  gsl_matrix_set_zero(G);
  gsl_matrix_set_zero(G_sum);

}

// function will jump to a random point in the matrix
// and pull out contiguous ncols
void randomize_iteration() {
  
  size_t start_row, start_col;
  size_t nrows, ncols;
  size_t offset = rand() % (All_X->size2 - B); //random number generator
  
  start_row = 0;
  start_col = offset;
  nrows = All_X->size1;
  ncols = B;
  X_view = gsl_matrix_submatrix(All_X, start_row, start_col, nrows, ncols);
  
  X = &X_view.matrix;

}

void initialize_iteration(int i) {
  
  size_t start_row, start_col;
  size_t nrows, ncols;

  //We are assuming num_iterations is multiple of batch size  
  //TODO: this calculation might change if we add multiple files to the
  //dataset
  size_t offset = i%B;

  start_row = 0;
  start_col = offset*B;
  nrows=All_X->size1;
  ncols = B;
  
  //cout<<"iteration "<<i<<", offset="<<offset<<", start_col="<<start_col<<endl;
  X_view = gsl_matrix_submatrix(All_X, start_row, start_col, nrows, ncols);
  
  X = &X_view.matrix;
  
}

void initialize_schedule() {
  R = gsl_matrix_alloc(L,B);
  initialize_computation_vars();
}

void cleanup_schedule() {
  gsl_matrix_free(R);
  free_computation_vars();
}

void cleanup_stuff() {
  cout<<"Cleaning stuff"<<endl;
  gsl_vector_free(A_new);
  gsl_vector_free(A_old);
  gsl_matrix_free(A);
  gsl_matrix_free(All_X);
}


// note that x is a single column of X
void compute_T(const Matrix D, const Matrix x) {
  //cout<<"\tComputing T"<<endl;
  //cout<<"before regularized stuff"<<endl;
  // T = (Dt * D + lambda*I) ^-1 * Dt * x
 
  gsl_matrix_set_identity(regularized);
  
  gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		 1.0, D, D,
		 lambda, regularized); //compute Dt*D+lambda*I

  // compute inverse of matrix
  gsl_matrix_memcpy(temp_LU,regularized);
  
  int signum;
  
  gsl_linalg_LU_decomp(temp_LU, p, &signum);
  
  gsl_linalg_LU_invert(temp_LU, p, regularized_inverse);
  //finally the inverse is ready

  //Matrix Dtx = gsl_matrix_alloc(K,1);
  //cout<<"\t before Dtx.."<<endl;
  gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		 1.0, D, x,
		 0.0, Dtx); //compute Dt*x

  //cout<<"\t before final mult.."<<endl;
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, regularized_inverse, Dtx,
		 0.0, T); //final multiplication
  
  
}

void compute_mycoefficients(Matrix S) {

 
 gsl_matrix_set_zero(D);
 Matrix x = gsl_matrix_alloc(L,1);
 vector<int> indices;
 indices.resize(K);
 solve_time=0;
 distance_time=0;
 
 for (int i=0; i<X->size2; i++) {  // for each column of X
   
   for (int j=0;j<K;j++)
     indices[j]=-1;

   util::copyColumn(x, X, i);
   
   start5=clock();
   util::return_K_min_indices(x, A, K, indices);
   finish5=clock();
   distance_time+=(double(finish5)-double(start5))/CLOCKS_PER_SEC;

   util::selectColumns(D, A, &indices[0], K);

   start4=clock();
   compute_T(D, x);
   finish4=clock();
   solve_time+=(double(finish4)-double(start4))/CLOCKS_PER_SEC;
   
   util::setColumnSelection(S, T, &indices[0], i, K);
    
 } // loop over all images


 indices.clear();
 gsl_matrix_free(x);
 

}

void compute_coefficients(Matrix S) {
  
  //  cout<<"\tComputing coefficients"<<endl;

  // need to do conversion here between gsl matrix A and flann matrix
  // flann is essentially expecting a transpose of the matrix
  Matrix At=gsl_matrix_alloc(M,L);
  gsl_matrix_transpose_memcpy(At,A);
  flann::Matrix<double> Af(At->data,M,L);

  flann::Index<double> A_index(Af, flann::AutotunedIndexParams(0.99, 0.1, 0.1, 0.8));
  A_index.buildIndex();

  /*
  // now create the flann index
  flann::Index<double> A_index(Af, flann::KDTreeIndexParams(16));
  A_index.buildIndex();
  */
  
  
  //Matrix D=gsl_matrix_alloc(L,K);
  gsl_matrix_set_zero(D);
  //gsl_vector *col_vector = gsl_vector_alloc(L);

  Matrix Xt=gsl_matrix_alloc(B,L);
  gsl_matrix_transpose_memcpy(Xt,X);
  
  //cout<<"\t\tPreparing knn queries"<<endl;
  flann::Matrix<double> query(Xt->data,B,L); // fill this with all columns
  flann::Matrix<int> indices(new int[query.rows*K], query.rows, K);
  flann::Matrix<float> dists(new float[query.rows*K], query.rows, K);

  A_index.knnSearch(query, indices, dists, K, flann::SearchParams(-2));

  //  A_index.knnSearch(query, indices, dists, K, flann::SearchParams(256));
  //util::printFlannIndices(indices,X->size2,K);

  Matrix x = gsl_matrix_alloc(L,1);
  
  for (int i=0; i<X->size2; i++) {  // for each column of X
    //if (i%2000==0)
    //  cout<<(double)i*100.0/X->size2<<"%, ";

    //cout<<"\t\t\tselectColumns"<<endl;
    util::selectColumns(D, A, indices[i], K);
    
    //cout<<"\t\t\tcopyColumn"<<endl;
    util::copyColumn(x, X, i);

    //cout<<"\t\t\tcompute T"<<endl;
    compute_T(D, x);
    
    //cout<<"\t\t\tsetcolumn selection"<<endl;
    //update the relevant column of S
    util::setColumnSelection(S, T, indices[i], i, K);
    
  } // loop over all images

  dists.free(); 
  indices.free();
  query.free();
  // A_index.free(); todo: check if this cleans up properly
  //Af.free();
  
  //todo: some of this can be freed up earlier
  //cout<<"Done with loop over all images"<<endl;

  //gsl_matrix_free(T);
  gsl_matrix_free(x);
  //gsl_matrix_free(D);
  //gsl_vector_free(col_vector);
  
  gsl_matrix_free(At); 

  //gsl_matrix_free(Xt); //results in segfault used by query
      
  //  cout<<"done w/ compute coefficients function"<<endl;
}

void compute_local_gradient(Matrix G,
			    const Matrix X,
			    const Matrix A,
			    const Matrix S) {
  
  //  cout<<"Computing local gradient"<<endl;
  //R = X - A*S;
  //  cout<<"R size is "<<R->size1<<"x"<<R->size2<<endl;
  gsl_matrix_memcpy(R,X); // first we set R=X

  //  cout<<"\t dgemm1"<<endl;
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 -1.0, A, S,
		 1.0, R); // and then do R - A*S
  
  // compute snr ratio here
  double x_sum,r_sum;
  //cout<<"computng snr"<<endl;
  util::sumMatrixElementSquares(X,x_sum);
  util::sumMatrixElementSquares(R,r_sum);
  double snr=10*log10(x_sum/r_sum);

  double alpha=1.0/(B*mpi_size);
  double fval = 0.5*alpha*r_sum;
  double fval2 = 0.0;
  logstring<<" snr="<<snr;
  //logstring<<" fval="<<fval;

  //cout<<"Computing local gradient"<<endl;
  // dG = -  R * St / B;  
  
  //cout<<"\t dgemm2"<<endl;
  gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		 -1.0*alpha, R, S,
		 0.0, G);

  // now we add the new term corresponding to neighbors
  //cout<<"\t new term"<<endl;
  double g_lm;
  double x_lb,a_lm,s_mb;
  
  for (int l=0; l<L; l++) {
   //if (L%100==0)
   // cout<<"L="<<l<<endl;
  
  gsl_vector_view A_lm = gsl_matrix_row(A,l);
    gsl_vector_view G_lm = gsl_matrix_row(G,l);
    gsl_vector_view X_lb = gsl_matrix_row(X,l);

    for (int m=0; m<M; m++) {
      
      //G_lm = gsl_matrix_get(G,l,m);
      //A_lm = gsl_matrix_get(A,l,m);
      g_lm = gsl_vector_get(&G_lm.vector,m);
      a_lm = gsl_vector_get(&A_lm.vector,m);
      gsl_vector_view S_mb = gsl_matrix_row(S,m);
      
      for (int b=0; b<B; b++) {
	//X_lb = gsl_matrix_get(X,l,b);
	//S_mb = fabs(gsl_matrix_get(S,m,b));
	
	s_mb=gsl_vector_get(&S_mb.vector,b);
	
	//if (s_mb<1e-5)
	//  s_mb = 0.01; //TODO:: hack
	
	g_lm = g_lm -  2*beta*(x_lb-a_lm)*fabs(s_mb);

	fval2 += beta*(x_lb-a_lm)*(x_lb-a_lm)*fabs(s_mb);
	
	gsl_matrix_set(G,l,m,g_lm);
      }
    }
  }

  //logstring<<" fval2="<<fval2;
  logstring<<" fsum="<<fval+fval2;

}


void compute_gradient_update(Matrix G) {
  
  //  cout<<"Computing gradient update"<<endl;
  Matrix S = gsl_matrix_alloc(M,B);
  Matrix E = gsl_matrix_alloc(L,B);
  gsl_matrix_set_zero(S);
  gsl_matrix_set_zero(E);
  
  //  compute_coefficients(S); //trashing the flann stuff for now
  
  start0=clock();
  compute_mycoefficients(S);
  finish0=clock();
  
  start1=clock();
  compute_local_gradient(G, X, A, S);
  finish1 = clock();

  coefficient_time=(double(finish0)-double(start0))/CLOCKS_PER_SEC;
  gradient_time=(double(finish1)-double(start1))/CLOCKS_PER_SEC;

  gsl_matrix_free(E);
  gsl_matrix_free(S);
}


void update_basis(Matrix G) {
  //  cout<<"Updating basis"<<endl;
  //A = A - eta*G; // need scalar matrix multiplication for eta
  start2=clock();
  util::matrixColsToVector(A,A_old);
  
  gsl_matrix_scale(G,-1.0*eta);
  gsl_matrix_add(A,G);

  util::matrixColsToVector(A,A_new);

  //cout<<"\t ddot"<<endl;
  double A_dot;  
  gsl_blas_ddot(A_old,A_new,&A_dot);
  double A_old_length = util::lengthVector(A_old);
  double A_new_length = util::lengthVector(A_new);
  double angle=acos(A_dot/(A_old_length*A_new_length));
  
  logstring<<" angle="<<angle; 
  if (angle<target_angle)
    eta=eta*1.01;
  else
    eta=eta/1.01;

  logstring<<" eta="<<eta<<endl;
  //cout<<"before normalize columns"<<endl;
  //normalize_columns of A;
  util::normalizeMatrixCols(A);
  finish2=clock();
  update_time=(double(finish2)-double(start2))/CLOCKS_PER_SEC;
}


void do_nearnet_learning(int schedule, int num_iterations, 
			 bool sliding_batch) {
  
  initialize_schedule();
  
  for (int iteration=0;iteration<num_iterations;iteration++) {
    start3=clock();

    if (sliding_batch)
      initialize_iteration(iteration);
    else
      randomize_iteration();

    logstring<<"Schedule="<<schedule<<" Iteration="<<iteration<<" target_angle="<<target_angle<<" K="<<K;
    compute_gradient_update(G);

#ifdef PARALLEL
    // intercept all gradients and do a global reduction
    MPI_Allreduce(G->data, G_sum->data, G->size1*G->size2, 
		  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    gsl_matrix_memcpy(G_sum, G);
#endif
    
    update_basis(G_sum);
    
    finish3=clock();
    total_iteration_time=(double(finish3)-double(start3))/CLOCKS_PER_SEC;

    timestring<<" Coefficient_time="<< coefficient_time;
    timestring<<" Distance_time="<<distance_time;
    timestring<<" Solve_time="<<solve_time;
    timestring<<" Gradient_time="<<gradient_time;
    timestring<<" Update_time="<<update_time;
    timestring<<" Total_time="<<total_iteration_time<<endl;
    
    if (mpi_rank==0) {
      logfile<<logstring.str();
      cout<<logstring.str();
      if (iteration%write_every_n_time==0){
	util::save_matrix_to_file(A, schedule, iteration, prefix);
        logfile.flush();
      }
      timefile<<timestring.str();
      timefile.flush();
    }
    
    logstring.clear();
    logstring.str("");
    timestring.clear();
    timestring.str("");

  } //iteration loop
  
  cleanup_schedule();

}

void test_noflann() {
  // test config restart
  //
  cout<<"TEST CASE!!!"<<endl;
  bool sliding_batch = false;
  int schedule=0;
  
  B=10;  K=75;  num_iterations=20000; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=20000; target_angle =0.075;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=20000; target_angle =0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=20000; target_angle =0.025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=40000; target_angle =0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=60000; target_angle =0.005;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=60000; target_angle =0.0025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=75;  num_iterations=80000; target_angle =0.001;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

}

void gaia3() {

  bool sliding_batch = false;
  int schedule=0;
  // schedule for gaia2:: IMAGES
  cout<<"gaia3"<<endl;
  B=10;  K=100;  num_iterations=15000; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=15000; target_angle =0.075;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=15000; target_angle =0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=25000; target_angle =0.025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=25000; target_angle =0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=40000; target_angle =0.005;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=40000; target_angle =0.0025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=60000; target_angle =0.001;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

}

void luna1() {
  bool sliding_batch = false;
  int schedule=0;

 // Burn-in phase
  /*
  // luna1
  B=10; K=750; num_iterations=50000; target_angle=0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
  
  B=10; K=500; num_iterations=50000; target_angle=0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10; K=250; num_iterations=50000; target_angle=0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
  */
  

}

void sol1() {

 cout<<"sol1"<<endl;
  bool sliding_batch = false;
  int schedule=0;
  B=10;  K=100;  num_iterations=10000; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=10000; target_angle =0.075;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=10000; target_angle =0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=20000; target_angle =0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=20000; target_angle =0.005;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=20000; target_angle =0.0025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=40000; target_angle =0.001;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

}


void sol2() {

 cout<<"sol2"<<endl;
  bool sliding_batch = false;
  int schedule=0;
  B=10;  K=100;  num_iterations=10000; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=10000; target_angle =0.075;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=10000; target_angle =0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=20000; target_angle =0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=20000; target_angle =0.005;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=20000; target_angle =0.0025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10;  K=100;  num_iterations=40000; target_angle =0.001;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

}



void terra2() {
  bool sliding_batch = false;
  int schedule=0;

 // Burn-in phase
  cout<<"terra2"<<endl;
  
  B=10; K=1000; num_iterations=1000; target_angle=0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
  
  B=10; K=1000; num_iterations=1000; target_angle=0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10; K=1000; num_iterations=1000; target_angle=0.025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=10; K=1000; num_iterations=2000; target_angle=0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

}

void hero3() {
  bool sliding_batch = false;
  int schedule=0;

  if (mpi_rank==0)
    cout<<"hero3"<<endl;
  
  B=1; K=1000; num_iterations=1000; target_angle=0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
  
  B=1; K=1000; num_iterations=1000; target_angle=0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=1; K=1000; num_iterations=1000; target_angle=0.025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=1; K=1000; num_iterations=2000; target_angle=0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=1; K=750; num_iterations=2000; target_angle=0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

}

void jack() {

 bool sliding_batch = false;
  int schedule=0;
  cout<<"jack"<<endl;

 B=10; K=400; num_iterations=1; target_angle=0.1;
 do_nearnet_learning(schedule++, num_iterations, sliding_batch);

/*
  B=10;  K=3072;  num_iterations=10; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  lambda=lambda/10; beta=beta/10;
  B=10;  K=3072;  num_iterations=10; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  lambda=lambda*10; beta=beta*10;
  B=10;  K=2000;  num_iterations=10; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  lambda=lambda/10; beta=beta/10;
  B=10;  K=2000;  num_iterations=10; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  lambda=lambda*10; beta=beta*10;
  B=10;  K=1000;  num_iterations=10; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  lambda=lambda/10; beta=beta/10;
  B=10;  K=1000;  num_iterations=10; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
*/
}

void nova1() {
  bool sliding_batch = false;
  int schedule=0;
  cout<<"nova1"<<endl;

  B=1;  K=1500;  num_iterations=1000; target_angle =0.1;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=1;  K=1000;  num_iterations=1000; target_angle =0.05;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=1; K=750; num_iterations=1000; target_angle=0.025;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
 
  
  /*
  B=20; K=20; num_iterations=10000; target_angle=0.005;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  B=20; K=20; num_iterations=10000; target_angle=0.001;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);

  sliding_batch = true;
  B=B_save; K=20; num_iterations=1000; target_angle=0.01;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
  
  B=B_save; K=250; num_iterations=1000; target_angle=0.005;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
  
  B=B_save; K=250; num_iterations=1000; target_angle=0.001;
  do_nearnet_learning(schedule++, num_iterations, sliding_batch);
   */
}

int main(int argc, char** argv) {

  if (argc<2) {
    cout<<"Run program [-restart] configuration_file"<<endl;
    exit(0);
  }

  string config_filename;

  if (argc==3) {
    string str=string(argv[2]);
    //string pattern="restart";
    //const char* pch=strstr(str.c_str(),pattern.c_str());
    restart = true;
    config_filename=string(argv[2]); 
  }

  if (argc==2) {
    config_filename=string(argv[1]);
  }


#ifdef PARALLEL
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
#endif
  cout<<"mpi rank= "<<mpi_rank<<", mpi_size= "<<mpi_size<<endl;

  
  string log_filename;
  logstring<<"Reading config file "<<config_filename<<endl;
  cout<<"Reading config file "<<config_filename<<endl;

  if (restart)
    util::read_restartfile(config_filename,
			   run_prefix, log_filename,
			   dataset_list,
			   A_initfile,
			   L, M,
			   NUM_IMAGES,write_every_n_time);
  else
    util::read_configfile(config_filename,
			  run_prefix,log_filename,
			  dataset_list,
			  L, M,
			  NUM_IMAGES,write_every_n_time);
  
  if (mpi_rank==0){
    logfile.open(log_filename.c_str());
    if (!logfile.is_open()){
      cout<<"Could not open"<<log_filename<<endl;
      exit(0);
    }
    else {
      //cout<<"Succeeded in opening logfile"<<endl;
    }
    string time_filename=log_filename+".time";
    timefile.open(time_filename.c_str());
    if (!timefile.is_open()){
      cout<<"Could not open"<<time_filename<<endl;
      exit(0);
    }
    else {
      //cout<<"Succeeded in opening timefile"<<endl;
    }


  }
  
  util::initialize_datasets(dataset,dataset_list,mpi_rank,mpi_size);

  initialize_stuff();
  if (mpi_rank==0) {
    logfile<<logstring.str();
    timefile<<timestring.str();
    cout<<logstring.str();
  }

  logstring.clear();
  logstring.str("");
  timestring.clear();
  timestring.str("");

  prefix=run_prefix+"-Iteration";
  bool sliding_batch=false;
  int schedule=0;

  //nova1();
  //sol2();
  //gaia3();
  //test_noflann();
  //terra2();
  //hero3();
  jack();

  gsl_matrix_free(G);
  gsl_matrix_free(G_sum);

  cleanup_stuff();

  logfile.flush();
  logfile.close();
  timefile.flush();
  timefile.close();


#ifdef PARALLEL
  MPI_Finalize();
#endif

  return 0;
}
