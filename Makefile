# Makefile for compiling GameOfLifeGFLW.cpp with GLFW library

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++11 -Wall -Wextra

# Directory containing GLFW library
GLFW_DIR := lib-arm64

# GLFW library
GLFW_LIB := $(GLFW_DIR)/libglfw3.a

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
	g++ -std=c++11 -Wall -Wextra GameOfLifeGFLW.o lib-arm64/libglfw3.a -o GameOfLifeGFLW -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo

# Clean up
clean:
	rm -f $(OBJS) $(EXEC)


