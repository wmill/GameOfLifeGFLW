#include <iostream>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <random>
#include <thread>
#include <string.h>
#include <stdio.h>

const int SCREEN_WIDTH = 1500;
const int SCREEN_HEIGHT = 1000;
const int IMAGE_WIDTH = SCREEN_WIDTH;
const int IMAGE_HEIGHT = SCREEN_HEIGHT;
const int NUM_THREADS = 16;
const int USE_GOOD_RANDOM = true;

typedef uint32_t Pixel;

const Pixel LIVE = 0xFF000000;
const Pixel DEAD = 0xFFFFFFFF;

__device__ Pixel getCell(Pixel* imageData, int h, int w) {
    return imageData[(h + IMAGE_HEIGHT) % IMAGE_HEIGHT * IMAGE_WIDTH + (w + IMAGE_WIDTH) % IMAGE_WIDTH];
}

__device__ int countNeighbors(Pixel* imageData, int h, int w) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            count += getCell(imageData, h + i, w + j) == LIVE ? 1 : 0;
        }
    }
    return count;
}

__global__ void updateImageKernel(Pixel* greenImageData, Pixel* redImageData) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("idx: %d\n", idx);
    if (idx >= IMAGE_WIDTH * IMAGE_HEIGHT) return;

    int h = idx / IMAGE_WIDTH;
    int w = idx % IMAGE_WIDTH;
    int neighbors = countNeighbors(greenImageData, h, w);
    Pixel cell = getCell(greenImageData, h, w);
    
    if (cell == LIVE) {
        if (neighbors < 2 || neighbors > 3) {
            redImageData[h * IMAGE_WIDTH + w] = DEAD;
        } else {
            redImageData[h * IMAGE_WIDTH + w] = LIVE;
        }
    } else {
        if (neighbors == 3) {
            redImageData[h * IMAGE_WIDTH + w] = LIVE;
        } else {
            redImageData[h * IMAGE_WIDTH + w] = DEAD;
        }
    }
    if (idx == 0)  {
        printf("redImageData[0]: %08X\ngreenImageData[0]: %08X\n", redImageData[0], greenImageData[0]);
    }
}

void randomizeImage(Pixel* imageData) {
    static std::random_device rd;
    static std::mt19937 eng(rd());
    static std::uniform_int_distribution<> distr(0, 1);
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int pixelIndex = y * IMAGE_WIDTH + x;
            imageData[pixelIndex] = distr(eng) == 0 ? LIVE : DEAD;
        }
    }
}

void fillHorizontalLines(Pixel* imageData) {
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int pixelIndex = y * IMAGE_WIDTH + x;
            imageData[pixelIndex] = y % 3 == 0 ? LIVE : DEAD;
        }
    }
}

void fillVerticalLines(Pixel* imageData) {
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int pixelIndex = y * IMAGE_WIDTH + x;
            imageData[pixelIndex] = x % 3 == 0 ? LIVE : DEAD;
        }
    }
}

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkOpenGLError(const char* message) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << message << ": OpenGL error " << err << std::endl;
    }
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "OpenGL Image Processing", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    Pixel* imageDataA = new Pixel[IMAGE_WIDTH * IMAGE_HEIGHT];
    Pixel* imageDataB = new Pixel[IMAGE_WIDTH * IMAGE_HEIGHT];
    memset(imageDataA, LIVE, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel));
    memset(imageDataB, DEAD, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel));

    // randomizeImage(imageDataA);
    fillHorizontalLines(imageDataA);
    fillVerticalLines(imageDataB);    

    Pixel *d_imageDataA, *d_imageDataB;
    cudaMalloc(&d_imageDataA, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel));
    cudaMalloc(&d_imageDataB, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel));
    cudaMemcpy(d_imageDataA, imageDataA, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imageDataB, imageDataB, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel), cudaMemcpyHostToDevice);

    cudaGraphicsResource *cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageDataA);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    while (!glfwWindowShouldClose(window)) {
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((IMAGE_WIDTH * IMAGE_HEIGHT + threadsPerBlock.x - 1) / threadsPerBlock.x);

        // Update the image data on the GPU
        updateImageKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imageDataA, d_imageDataB);
        cudaDeviceSynchronize();
        std::swap(d_imageDataA, d_imageDataB);

        // Map the CUDA resource and copy data to OpenGL texture
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaArray_t cudaArray;
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);
        cudaMemcpyToArray(cudaArray, 0, 0, d_imageDataA, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaResource, 0);

        // Render the updated texture
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(window);
        glfwPollEvents();

        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }

    cudaGraphicsUnregisterResource(cudaResource);
    cudaFree(d_imageDataA);
    cudaFree(d_imageDataB);
    delete[] imageDataA;
    delete[] imageDataB;

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
