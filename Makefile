# Makefile for compiling GameOfLifeGFLW.cpp with GLFW library and GameOfLifeGFLW.cu with nvcc

# Compiler for C++
CXX := g++

# Compiler for CUDA
NVCC := nvcc

# Compiler flags for C++
CXXFLAGS := -std=c++11 -Wall -Wextra

# Compiler flags for CUDA
NVCCFLAGS := 

# Executable names
EXEC_CPP := GameOfLifeGFLW
EXEC_CU := GameOfLifeGFLW-cuda

# Source files
SRCS_CPP := GameOfLifeGFLW.cpp
SRCS_CU := GameOfLifeGFLW-cuda.cu

# Object files
OBJS_CPP := $(SRCS_CPP:.cpp=.o)
OBJS_CU := $(SRCS_CU:.cu=.o)

# Default target
all: $(EXEC_CPP) $(EXEC_CU)

# Compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link C++ object files and GLFW library into executable
$(EXEC_CPP): $(OBJS_CPP)
	$(CXX) $(CXXFLAGS) $(OBJS_CPP) -o $(EXEC_CPP) -lglfw -lX11 -lXrandr -lXi -lXinerama -lXcursor -lrt -lm -ldl -lpthread -lGL -lGLU

# Link CUDA object files and GLFW library into executable
$(EXEC_CU): $(OBJS_CU)
	$(NVCC) $(NVCCFLAGS) $(OBJS_CU) -o $(EXEC_CU) -lglfw -lGL -lGLU

# Clean up
clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(EXEC_CPP) $(EXEC_CU)
