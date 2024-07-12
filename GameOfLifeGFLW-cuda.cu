#include <iostream>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <random>
#include <string.h>

const int SCREEN_WIDTH = 2000;
const int SCREEN_HEIGHT = 1000;
const double PIXEL_RATIO = 0.1;
const int IMAGE_WIDTH = SCREEN_WIDTH / PIXEL_RATIO;
const int IMAGE_HEIGHT = SCREEN_HEIGHT / PIXEL_RATIO;


typedef uint32_t Pixel;

const Pixel CELL_LIVE = 0xFF000000;
const Pixel CELL_DEAD = 0xFFFFFFFF;
const int CELL_FRACTION_ALIVE = 10; // 1 in CELL_FRACTION_ALIVE cells will be alive for random initialization

const size_t BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel);
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

__device__ Pixel getCell(Pixel* imageData, int h, int w) {
    return imageData[(h + IMAGE_HEIGHT) % IMAGE_HEIGHT * IMAGE_WIDTH + (w + IMAGE_WIDTH) % IMAGE_WIDTH];
}

__device__ int countNeighbors(Pixel* imageData, int h, int w) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            count += getCell(imageData, h + i, w + j) == CELL_LIVE ? 1 : 0;
        }
    }
    return count;
}

__global__ void updateImageKernel(Pixel* greenImageData, Pixel* redImageData) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IMAGE_SIZE) return;

    int h = idx / IMAGE_WIDTH;
    int w = idx % IMAGE_WIDTH;
    int neighbors = countNeighbors(greenImageData, h, w);
    Pixel cell = getCell(greenImageData, h, w);
    
    if (cell == CELL_LIVE) {
        if (neighbors < 2 || neighbors > 3) {
            redImageData[h * IMAGE_WIDTH + w] = CELL_DEAD;
        } else {
            redImageData[h * IMAGE_WIDTH + w] = CELL_LIVE;
        }
    } else {
        if (neighbors == 3) {
            redImageData[h * IMAGE_WIDTH + w] = CELL_LIVE;
        } else {
            redImageData[h * IMAGE_WIDTH + w] = CELL_DEAD;
        }
    }
}

void randomizeImage(Pixel* imageData) {
    static std::random_device rd;
    static std::mt19937 eng(rd());
    static std::uniform_int_distribution<> distr(0, CELL_FRACTION_ALIVE - 1);
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int pixelIndex = y * IMAGE_WIDTH + x;
            imageData[pixelIndex] = distr(eng) == 0 ? CELL_LIVE : CELL_DEAD;
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

    // Set the texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // the following three lines are technically not necessary. They affect non 1:1 pixel to texel mapping and offscreen textures.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate texture memory (important: use GL_RGBA8 for internal format)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    Pixel* imageData = new Pixel[IMAGE_SIZE];
    memset(imageData, CELL_LIVE, BUFFER_SIZE);

    randomizeImage(imageData);

    Pixel *d_imageDataA, *d_imageDataB;
    checkCudaError(cudaMalloc(&d_imageDataA, BUFFER_SIZE), "Failed to allocate device memory for d_imageDataA");
    checkCudaError(cudaMalloc(&d_imageDataB, BUFFER_SIZE), "Failed to allocate device memory for d_imageDataB");
    checkCudaError(cudaMemcpy(d_imageDataA, imageData, BUFFER_SIZE, cudaMemcpyHostToDevice), "Failed to copy imageData to device memory");

    cudaGraphicsResource *cudaResource;
    checkCudaError(cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard), "Failed to register GL texture with CUDA");

    while (!glfwWindowShouldClose(window)) {
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((IMAGE_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);

        updateImageKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imageDataA, d_imageDataB);
        checkCudaError(cudaDeviceSynchronize(), "CUDA kernel execution failed");
        std::swap(d_imageDataA, d_imageDataB);

        // Map the CUDA resource and copy data to OpenGL texture
        checkCudaError(cudaGraphicsMapResources(1, &cudaResource, 0), "Failed to map CUDA graphics resource");
        cudaArray_t cudaArray;
        checkCudaError(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0), "Failed to get CUDA array from mapped resource");
        checkCudaError(cudaMemcpyToArray(cudaArray, 0, 0, d_imageDataA, BUFFER_SIZE, cudaMemcpyDeviceToDevice), "Failed to copy data to CUDA array");
        checkCudaError(cudaGraphicsUnmapResources(1, &cudaResource, 0), "Failed to unmap CUDA graphics resource");

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

        checkOpenGLError("Rendering loop");
    }

    checkCudaError(cudaGraphicsUnregisterResource(cudaResource), "Failed to unregister CUDA graphics resource");
    checkCudaError(cudaFree(d_imageDataA), "Failed to free device memory for d_imageDataA");
    checkCudaError(cudaFree(d_imageDataB), "Failed to free device memory for d_imageDataB");
    delete[] imageData;

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
