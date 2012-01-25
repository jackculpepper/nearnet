
#ifndef _UTIL_H_
#define _UTIL_H_

#include <flann/flann.hpp>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

typedef gsl_matrix* Matrix;

using namespace std;
class util
{
 public:
  static void read_uint8_matrix_from_file(Matrix m, const string filename);
  static void read_double_matrix_from_file(Matrix m, const string filename);

  //  static Matrix read_matrix_from_file(const string filename);

  static void read_restartfile(string filename,
			       string &prefix, string &logfile,
			       string &dataset_list,
			       string &restart_file,
			       size_t &L, size_t& M, 
			       size_t &num_images, int& write_interval);

  static void read_configfile(string filename,
			      string &prefix, string &logfile,
			      string &dataset_list,
			      size_t &L, size_t& M, 
			      size_t &num_images, int& write_interval);

  static void initialize_datasets(string &dataset,
				  const string dataset_list,
				  const int mpi_rank,
				  const int mpi_size);
  

  static  void return_K_min_indices(const Matrix x,
					  const Matrix A,
					  const int K,
					  vector<int> &indices);
				   

  static void save_matrix_to_file(const Matrix m, const int &schedule, const int &i, const string &prefix);
  
  static void print_matrix(const Matrix m, const string &title);
  static void print_matrix_pieces(const Matrix m, const string &title);

  static  double lengthVector( gsl_vector * vector );
  static  void normalizeVector( gsl_vector * vector );
  
  static  void normalizeMatrixCols(Matrix matrix );

  static  void selectColumns(Matrix target, const Matrix src, const int* indices, const int nn);
  
  static  void copyColumn(Matrix target, const Matrix src, const int col);
  
  static  void setColumnSelection(Matrix target, const Matrix src, const int* indices, const int column, const int K);

  static  void matrixColsToVector(gsl_matrix *, gsl_vector * );

  static void standardizeImages(Matrix m);

  static void removeMeanFromImages(Matrix m);
  
  static void randomize(Matrix m);
  
  static  void shuffleMatrixColumns(Matrix src);

  static void sumMatrixElements(Matrix m, double& total);
  
  static void sumMatrixElementSquares(Matrix m, double& total);

  static bool matrixEqual(Matrix m, Matrix n);

  static void printFlannIndices(flann::Matrix<int> m, int numImages, int knn);
  
  static void check_matrix(Matrix m, string msg);

  static void clean_matrix(Matrix m);
};

#endif
