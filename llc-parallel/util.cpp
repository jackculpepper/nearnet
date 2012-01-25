#include "util.h"

void util::read_configfile(string filename,
			   string &prefix, string &logfile,
			   string &dataset_list,
			   size_t &L, size_t& M, 
			   size_t& num_images, int& write_interval) {

  //cout<<"Trying to open config file "<<filename<<endl;
  ifstream config(filename.c_str());
  if (config.is_open()) {
    string name;
    string val;
    int ival;
    config>>name>>prefix;
    config>>name>>logfile;
    config>>name>>dataset_list;
    config>>name>>L;
    config>>name>>M;
    config>>name>>num_images;
    config>>name>>write_interval;
  }
  else {
    cout<<"Could not open "<<filename<<endl;
    exit(0);
  }
  config.close();
}

void util::read_restartfile(string filename,
			    string &prefix, string &logfile,
			    string &dataset_list,
			    string &restart_file,
			    size_t &L, size_t& M, 
			    size_t &num_images, int& write_interval) {
  
  
  //cout<<"Trying to open config file "<<filename<<endl;
  ifstream config(filename.c_str());
  if (config.is_open()) {
    string name;
    string val;
    int ival;
    config>>name>>prefix;
    config>>name>>logfile;
    config>>name>>dataset_list;    
    config>>name>>restart_file;
    config>>name>>L;
    config>>name>>M;
    config>>name>>num_images;
    config>>name>>write_interval;
  }
  else {
    cout<<"Could not open "<<filename<<endl;
    exit(0);
  }
  config.close();
}
// assuming that m is already allocated
void util::read_uint8_matrix_from_file(Matrix m,
				 const string filename) {
  
  FILE *f = fopen(filename.c_str(),"rb");
  if (f!=NULL) {
    //gsl_matrix_fread(f,m);
    int num_elements=m->size1*m->size2;
    //cout<<"num_elements "<<num_elements<<endl;
    char* buf=(char*)malloc(num_elements*sizeof(unsigned char));
    if (!buf) {
      cout<<"could not allocate buffer for reading data"<<endl;
      fclose(f);
      exit(0);
    }
    fread(buf,sizeof(unsigned char),num_elements,f);
    
    int k=0;
    for (int i=0;i<m->size1;i++){
      for (int j=0;j<m->size2;j++){
	gsl_matrix_set(m,i,j,(double)buf[k]);
	k++;
      }
    }
    free(buf);
    fclose(f);
  }
  else {
    cout<<"Could not read "<<filename<<endl;
    exit(0);
  }
}

void util::read_double_matrix_from_file(Matrix m,
				 const string filename) {
  
  FILE *f = fopen(filename.c_str(),"rb");
  if (f!=NULL) {
    gsl_matrix_fread(f,m);
    fclose(f);
  }
  else {
    cout<<"Could not read "<<filename<<endl;
    exit(0);
  }
}

/*
Matrix util::read_matrix_from_file(const string filename) {

  //TODO:: need to determine size1 and size2 by parsing filename
  int size1=0;
  int size2=0;
  Matrix m = gsl_matrix_alloc(size1, size2);

  FILE *f = fopen(filename.c_str(),"rb");
  if (f!=NULL) {
    gsl_matrix_fread(f,m);
    fclose(f);
  }
  else {
    cout<<"Warning: could not read data file "<<filename<<endl;
    exit(0);
  }

  check_matrix(m,"read matrix from file");
  return m;
}
*/

void util::save_matrix_to_file(const Matrix m, 
			       const int &schedule,
			       const int &iteration, 
			       const string &prefix) {
  
  // we need to write the size of the matrix
  // nrows
  // ncols
  // and then the rest of the data
  string filename;
  check_matrix(m,"save matrix to file");

  stringstream ss;
  ss<<prefix<<"-S-"<<setw(4)<<setfill('0')<<schedule<<"-I-";
  ss<<setw(6)<<setfill('0')<<iteration;
  filename=ss.str();
  stringstream rows,cols;
  rows<<"-L-"<<setw(6)<<setfill('0')<<m->size1;
  cols<<"-M-"<<setw(6)<<setfill('0')<<m->size2;
  filename=filename+"-"+rows.str()+"-"+cols.str()+".dat";
  
  FILE *f = fopen(filename.c_str(),"wb");
  if (f!=NULL) {
    gsl_matrix_fwrite(f,m);
    fclose(f);
  }
  else {
    cout<<"Warning:: could not open "<<filename<<" for writing"<<endl;
  }
  
}

void util::print_matrix(const Matrix m, 
		       const string &name){
  
  int nr=m->size1;
  int nc=m->size2;
  //printf("Printing matrix %s\n",name);
  
  int i=0;
  for (int r=0;r<nr;r++) {
    for (int c=0;c<nc;c++) {
      printf("%g,", gsl_matrix_get(m,r,c));
      
    }
    printf("\n");
  }
  
}

void util::print_matrix_pieces(const Matrix m, 
			       const string &name){
  
  int nr=m->size1;
  int nc=m->size2;
  cout<<"Printing random pieces of matrix %s\n"<<name<<endl;
  
  int num_pieces=10;
  int count=0;
  
  int i=0;
  for (int r=0;r<nr;r++) {

    double flip=rand()/RAND_MAX;
    if (flip<0.5) {
      for (int c=0;c<nc;c++) {
	cout<<gsl_matrix_get(m,r,c)<<", ";
      }
      count++;
    }
    cout<<endl;
    if (count>num_pieces)
      return;
  }
  
}
double util::lengthVector( gsl_vector *vector) {
  double norm=0.0;
  if(!gsl_vector_isnull(vector))
    {
      norm = cblas_dnrm2(vector->size,vector->data, 1);
    }  
  return norm;
}

/* Normalize Vector */
void util::normalizeVector( gsl_vector * vector )
{
  double norm;
  if(!gsl_vector_isnull(vector))
    {
      norm = cblas_dnrm2(vector->size,vector->data, 1);
      gsl_vector_scale(vector,(double)(1.0/norm));
    }  
}

/* Normalize Matrix Columns */
void util::normalizeMatrixCols( Matrix  matrix )
{
  check_matrix(matrix,"normalize matrix cols");

  int rows = matrix->size1;
  int cols = matrix->size2;
  int count;
  
  gsl_vector * tempVector = gsl_vector_alloc(rows);

  for(count = 0; count < cols; count++)
    {
      gsl_matrix_get_col(tempVector,matrix,count);
      normalizeVector(tempVector);
      gsl_matrix_set_col(matrix,count,tempVector);
    }

  /* Free up allocated memory */
  gsl_vector_free(tempVector);

}

// function will take all indices, look up columns in src
// and drop them into target.
// it is assumed that target is already allocated and of the right size
void util::selectColumns(Matrix target, const Matrix src, 
			 const int* indices, const int nn) {

  //check_matrix(target,"selectColumns target matrix");
  //check_matrix(src, "selectColumns src matrix");

  if (!indices) {
    cout<<"Warning:: selectColumns:: indices are null"<<endl;
    exit(0);
  }

  //gsl_matrix_set_zero(target);

  // pull out indices columns from src matrix and drop them into target
  for (int i=0;i<nn; i++) { // TODO: assert that
                            // nn==indices.rows
    
    int col = indices[i];
    gsl_vector_view column = gsl_matrix_column(src, col);

    /*
    if  (((&column.vector)->size)!=target->size1) {
      cout<<"column vector size does not match target column size"<<endl;
      cout<<"src column vector size="<<(&column.vector)->size<<endl;
      cout<<"target column size="<<target->size1<<endl;
    }
    */
    gsl_matrix_set_col(target, i, &column.vector);//TODO:: check??
    
  }

}

// this function will do the following
// target = src(:,column);
// target is assumed to be 1 column wide
void util::copyColumn(Matrix target, const Matrix src, const int col) {
  //check_matrix(target,"copyColumn target matrix");
  //check_matrix(src, "copyColumn src matrix");
  gsl_vector_view column = gsl_matrix_column(src, col);
  gsl_matrix_set_col(target, 0, &column.vector);//TODO:: check

}

// this function will do the following operation
// target(indices,b) = src(:)
// src matrix is Kx1 
void util::setColumnSelection(Matrix target,const Matrix src,
			      const int* indices, const int column, 
			      const int K) {
  //check_matrix(target,"setColumnSelection target matrix");
  //check_matrix(src,"setColumnSelection src matrix");

  for (int i=0;i<K;i++) {
    int row_index = indices[i];
    int value = gsl_matrix_get(src,i,0); // 0th column
    gsl_matrix_set(target, row_index, column, value);
  }

}


// function will look at vector x, and compute distance to all columns
// of X
// it will then return K indices corresponding to minimum distance to
// columns of columns in A 
void util::return_K_min_indices(const Matrix x,
				const Matrix A,
				const int K,
				vector<int> &indices) 
{
  int num_Acols = A->size2;
  gsl_vector* distance=gsl_vector_alloc(num_Acols);
  gsl_vector* diff=gsl_vector_alloc(A->size1);

  gsl_vector_view xx=gsl_matrix_column(x,0);
  
  for (int i=0;i<num_Acols;i++) {
    double d=0;
    gsl_vector_view a=gsl_matrix_column(A,i);
    
    gsl_vector_memcpy(diff,&a.vector);
    gsl_vector_sub(diff,&xx.vector);

    d = gsl_blas_dnrm2(diff);
    gsl_vector_set(distance, i, d);
    //cout<<d<<" ,";
  }
  //cout<<endl;

  vector<size_t> temp;
  temp.resize(K);

  gsl_sort_vector_smallest_index(&temp[0],K,distance);
  for (int i=0;i<K;i++) {
    indices[i]=temp[i];
    //cout<<indices[i]<<", ";
  }

  //cout<<endl;

  temp.clear();
  gsl_vector_free(diff);
  gsl_vector_free(distance);
}

void util::standardizeImages(Matrix m) {
  
  check_matrix(m,"standardizeImages");

  int length = m->size1;
  int noOfImages = m->size2;
  int count;
  double std;

  gsl_vector * tempVector = gsl_vector_alloc(length);
  
  for(count = 0; count < noOfImages; count++)
    {
      gsl_matrix_get_col(tempVector,m,count);
      std = gsl_stats_sd(tempVector->data,1,length);

      if (std<1e-5)
	cout<<"WARNING:: standardizeImages found std="<<std<<" for image "<<count<<endl;

      gsl_vector_scale(tempVector,(1.0/std));
      gsl_matrix_set_col(m,count,tempVector);
    }
  
  gsl_vector_free(tempVector);
  
}

void util::removeMeanFromImages(Matrix m) {

  check_matrix(m, "removeMeanFromImages");

  int length = m->size1;
  int noOfImages = m->size2;
  int count;
  double mean;

  gsl_vector * tempVector = gsl_vector_alloc(length);

  for(count = 0; count<noOfImages; count++)
    {
      gsl_matrix_get_col(tempVector,m,count);
      mean = gsl_stats_mean(tempVector->data,1,length);
      gsl_vector_add_constant(tempVector,(-1.0*mean));
      gsl_matrix_set_col(m,count,tempVector);
    }
  
  gsl_vector_free(tempVector);

}

// function will fill up random entries into matrix
void util::randomize(Matrix m) {
  int rows=m->size1;
  int cols=m->size2;
  for (int i=0;i<rows;i++) {
    for (int j=0;j<cols;j++) {
      
      double flip=((double)rand())/RAND_MAX;
      double val=((double)rand())/RAND_MAX;
      double sign=-1;
      if (flip>0.5)
	sign=+1;
      gsl_matrix_set(m,i,j,sign*val);
    }
  }
  
}

void util::sumMatrixElements(Matrix m, double& total) {
  total = 0;
  for (int count = 0; count < (m->size1*m->size2); count++) 
    total += m->data[count];
  
}

void util::sumMatrixElementSquares(Matrix m, double& total) {
  total = 0;
  //check_matrix(m,"dummy");
  for (int count = 0; count < (m->size1*m->size2); count++) {
    double val= m->data[count];
#ifdef DEBUG
    if (isnan(val)) {
      cout<<"WARNING: detected nan in matrix index="<<count<<endl;
      exit(0);
    }
#endif
    total += (val*val);
  }
}

bool util::matrixEqual(Matrix m, Matrix n) {
  bool retval=true;
  if (m->size1!=n->size1)
    return false;

  if (m->size2!=n->size2)
    return false;
  
  for (int i=0;i<m->size1;i++) {
    for (int j=0;j<m->size2;j++) {
      double mx = gsl_matrix_get(m,i,j);
      double nx = gsl_matrix_get(n,i,j);
      double diff=fabs(mx-nx);
      if (diff>1e-5) {
	cout<<"Detected difference "<<diff<<" m="<<mx<<", n="<<nx<<endl;
	return false;
      }
    }
  }
  return retval;
}

void util::printFlannIndices(flann::Matrix<int> m, int numImages, int knn) {

  cout<<endl;
  for (int i=0;i<numImages;i++) {
    cout<<"Image "<<i<<" knn indices are: ";
    int* indices=m[i];
    for (int k=0;k<knn;k++) {
      cout<<indices[k]<<", ";
    }
    cout<<endl;
  }
  cout<<endl;
}

void util::check_matrix(Matrix m, string msg) {
#ifdef DEBUG
  if (gsl_matrix_isnull(m)) {
    cout<<"Warning:: "<<msg<<" matrix is null"<<endl;
    exit(0);
  }
  if ((m->size1<1)||(m->size2<1)) {
    cout<<"Warning:: "<<msg<<" matrix is not properly sized"<<endl;
    exit(0);
  }
  //cout<<endl;
  for (int i=0;i<m->size1;i++) {
    for (int j=0;j<m->size2;j++) {
      double val = gsl_matrix_get(m,i,j);
      if (isnan(val)) {
	cout<<".";
      }
    }
  }
  //cout<<endl;
#endif
}

void util::clean_matrix(Matrix m) {
  for (int i=0;i<m->size1;i++) {
    for (int j=0;j<m->size2;j++) {
      double val = gsl_matrix_get(m,i,j);
      if (isnan(val)) {
	gsl_matrix_set(m,i,j,0.0);
	cout<<"WARNING:: input matrix has nan at ("<<i<<","<<j<<")"<<endl;
      }
    }
  }
}


/* Places the columns of a matrix into a vector */
void util::matrixColsToVector(gsl_matrix * matrixData, gsl_vector * vectorData )
{
  
  int frameSize = matrixData->size1;
  int noOfFrames = matrixData->size2;
  int row,col;
  
  for(col = 0; col < noOfFrames; col++) {
    for (row = 0; row < frameSize; row++) {
      /* frame vector into cols */
      gsl_vector_set(vectorData,(row+(frameSize*col)),gsl_matrix_get(matrixData,row,col));
      
    }
  }
}


void util::shuffleMatrixColumns(Matrix src) {

  size_t num_rows=src->size1;
  size_t num_cols=src->size2;

  gsl_vector *order = gsl_vector_alloc(num_cols);
  
  for (int i=0;i<num_cols;i++) {
    gsl_vector_set(order,i,(double)i);
  }

  gsl_rng_env_setup();
  gsl_rng* r = gsl_rng_alloc(gsl_rng_default);

  gsl_ran_shuffle(r, order->data, num_cols, sizeof(double));

  for (int c=0;c<num_cols;c++) {
    int target_c = (int)gsl_vector_get(order,c);
    gsl_matrix_swap_columns(src, c, target_c);
  }
  
  //gsl_rng_free(r);
  //gsl_vector_free(order);
 
}

//
// function will open the text file dataset_list
// look into the first mpi_size entries
// and set dataset=dataset_list[mpi_rank]
void util::initialize_datasets(string &dataset,
			       const string dataset_list,
			       const int mpi_rank,
			       const int mpi_size) {

  ifstream list(dataset_list.c_str());
  std::vector<string> filenames;
  string line;
  string dat_type=".dat";
  string tiny_type="tiny";
  const char *pch=0;
  const char *pch2=0;
  if (list.is_open()) {
    while (!list.eof()) {
      getline(list,line);
      pch = strstr(line.c_str(),dat_type.c_str());
      pch2 = strstr(line.c_str(),tiny_type.c_str());
      if ((pch)||(pch2))
	filenames.push_back(line);
    }
  }
  else {
    cout<<"Could not open "<<dataset_list<<endl;
    exit(0);
  }
  list.close();  
  /*
    if (mpi_rank==0){
    cout<<"Read "<<filenames.size()<<" entries"<<endl;
    for (int i=0;i<filenames.size();i++) {
    cout<<filenames[i]<<endl;
    }
    }
  */
  dataset=filenames[mpi_rank];
  cout<<"MPI task "<<mpi_rank<<" will process "<<dataset<<endl;
  filenames.clear();
}
