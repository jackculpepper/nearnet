
#FLANN = /global/u1/p/prabhat/euclid/vs265/project/flann-install
FLANN=/d/visguests/prabhat/courses/VS265/project/flann-install
GSL = /d/visguests/prabhat/courses/VS265/project/gsl-install
#ACML = /d/visguests/prabhat/courses/VS265/project/acml/gnu64
ACML = /d/visguests/prabhat/courses/VS265/project/acml-fortran
ATLAS = /usr/local/atlas
PROGRAM = project

INCLUDEDIRS = \
	-I$(FLANN)/include \
	-I$(GSL)/include


LIBDIRS = \
	-L$(FLANN)/lib \
	-L$(GSL)/lib \
        -L$(ATLAS)/lib

LIBS = -lflann -lgsl -lcblas -latlas
#-lgslcblas
#lacml -lacml_mv -lacml_mp
#-lgslcblas

CXXSOURCES = test.cpp util.cpp project.cpp
CXXOBJECTS = $(CXXSOURCES:.cpp=.o)  # expands to list of object files
CXXFLAGS = $(INCLUDEDIRS) -DDEBUG -DPARALLEL
CXX= mpicxx.openmpi -O3
#CXX = g++ -O3

LDFLAGS = $(LIBDIRS) $(LIBS)

all: $(PROGRAM)

$(PROGRAM): $(CXXOBJECTS)
	$(CXX) -o $@ $(CXXOBJECTS) $(LDFLAGS)


project.o: project.cpp 
	$(CXX) $(CXXFLAGS) -c -o project.o project.cpp

util.o: util.cpp 
	$(CXX) $(CXXFLAGS) -c -o util.o util.cpp

test.o: test.cpp 
	$(CXX) $(CXXFLAGS) -c -o test.o test.cpp

clean:
	$(RM) $(CXXOBJECTS) $(PROGRAM) 

