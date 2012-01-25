
#ifndef _TEST_H_
#define _TEST_H_

#include "util.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

typedef gsl_matrix* Matrix;

using namespace std;
class test
{
 public:
  static void start_tests();

 private:
  static void check_file_export_import();
  static void prepare_output_for_matlab();
  static void test_matlab_import();
};

#endif
