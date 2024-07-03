# Makefile for compiling GameOfLifeGFLW.cpp with GLFW library

# TODO: also build cuda version with nvcc
# current command is `nvcc -o GameOfLifeGFLW-cu GameOfLifeGFLW.cu -lglfw -lGL -lGLU`


# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++11 -Wall -Wextra

# Executable name
EXEC := GameOfLifeGFLW

# Source file
SRCS := GameOfLifeGFLW.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Default target
all: $(EXEC)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@



# Link object files and GLFW library into executable
# $(EXEC): $(OBJS)
# 	$(CXX) $(CXXFLAGS) $(OBJS) $(GLFW_LIB) -o $(EXEC)

GameOfLifeGFLW: GameOfLifeGFLW.o
	g++ -std=c++11 -Wall -Wextra GameOfLifeGFLW.o -o GameOfLifeGFLW -lglfw -lX11 -lXrandr -lXi -lXinerama -lXcursor -lrt -lm -ldl -lpthread -lGL -lGLU

# Clean up
clean:
	rm -f $(OBJS) $(EXEC)


