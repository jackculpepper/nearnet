#include "test.h"

void test::start_tests() {
  //check_file_export_import();
  //prepare_output_for_matlab();
  test_matlab_import();

}
void test::test_matlab_import() {
  cout<<"trying to allocate matrix"<<endl;
  Matrix m=gsl_matrix_alloc(7,6);
  cout<<"trying to read matrix"<<endl; 
  util::read_double_matrix_from_file(m,"matlab-output.dat");
  cout<<"trying to print matrix"<<endl;
  util::print_matrix(m,"matlab import");
  gsl_matrix_free(m);

}

void test::prepare_output_for_matlab() {
  Matrix m=gsl_matrix_alloc(3,5);
  for (int i=0;i<3;i++) {
    for (int j=0;j<5;j++) {
      double val=i*10+j;
      gsl_matrix_set(m,i,j,val);
    }
  }
  util::print_matrix(m,"m");
  util::save_matrix_to_file(m, 0, 0, "test-matlab");

}

void test::check_file_export_import() {
  Matrix m=gsl_matrix_alloc(3,5);
  util::randomize(m);
  
  cout<<"Trying to save matrix to file"<<endl;
  util::save_matrix_to_file(m, 0, 0, "test-read-write");
  
  util::print_matrix(m,"m");
  
  cout<<"Trying to read matrix from file"<<endl;
  Matrix n=gsl_matrix_alloc(3,5);
  util::read_double_matrix_from_file(n,"test-read-write00000.dat");
  //Matrix n =util::read_matrix_from_file( "test-read-write00000.dat");
  //util::print_matrix(n,"n");
  
  cout<<"Checking for matrix equality"<<endl;
  bool equal=util::matrixEqual(m,n);
  if (equal) 
    cout<<"Read/Write Test succeeded"<<endl;
  else
    cout<<"Read/Write Test failed"<<endl;
  
}

