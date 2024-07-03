#include <iostream>

#include <GLFW/glfw3.h>

#include <random>
#include <thread>
#include <string.h>



const int SCREEN_WIDTH = 1500;
const int SCREEN_HEIGHT = 1000;
const int IMAGE_WIDTH = SCREEN_WIDTH;  // Change according to your image size
const int IMAGE_HEIGHT = SCREEN_HEIGHT;  // Change according to your image size
const int NUM_THREADS = 16;
const int USE_GOOD_RANDOM = true;

typedef uint32_t Pixel;

const Pixel LIVE = 0xFF000000;
const Pixel DEAD = 0xFFFFFFFF;

void randomizeImage(Pixel* imageData) {
    // Create random number generator
    static std::random_device rd;  // Obtain a random seed from hardware
    static std::mt19937 eng(rd()); // Standard mersenne_twister_engine seeded with rd()

    // Define the distribution
    static std::uniform_int_distribution<> distr(0, 4); // Range [0,1]
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int pixelIndex = y * IMAGE_WIDTH + x;
            // Example: Fill with random values LIVE or DEAD
            if (USE_GOOD_RANDOM) {
				imageData[pixelIndex] = distr(eng) == Pixel(0) ? LIVE : DEAD;
			}
            else {
				imageData[pixelIndex] = (rand() % 2) == Pixel(0) ? LIVE : DEAD;
			}

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

Pixel getCell(Pixel* imageData, int h, int w) {
    return imageData[(h + IMAGE_HEIGHT) % IMAGE_HEIGHT * IMAGE_WIDTH + (w + IMAGE_WIDTH) % IMAGE_WIDTH];
}

int countNeighbors(Pixel* imageData, int h, int w) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;
            }
            count += getCell(imageData, h + i, w + j) == LIVE ? 1 : 0;
        }
    }
    return count;
}

void updateImage(Pixel* greenImageData, Pixel* redImageData) {
    for (int h = 0; h < IMAGE_HEIGHT; h++) {
        for (int w = 0; w < IMAGE_WIDTH; w++) {
            int neighbors = countNeighbors(greenImageData, h, w);
            Pixel cell = getCell(greenImageData, h, w);
            if (cell == LIVE) {
                if (neighbors < 2 || neighbors > 3) {
                    redImageData[h * IMAGE_WIDTH + w] = DEAD;
                }
                else {
                    redImageData[h * IMAGE_WIDTH + w] = LIVE;
                }
            }
            else {
                if (neighbors == 3) {
                    redImageData[h * IMAGE_WIDTH + w] = LIVE;
                }
                else {
                    redImageData[h * IMAGE_WIDTH + w] = DEAD;
                }
            }
        }
    }
    return;
}


void generateNextGenerationPartial(Pixel* greenImageData, Pixel* redImageData, int start, int end) {
    // start and end are the indexes of the pixels that the current thread is responsible for
    int maxIndex =  IMAGE_WIDTH * IMAGE_HEIGHT;
    if (end > maxIndex) {
		end = maxIndex;
	}
    for (int i = start; i < end; i++) {
        int h = i / IMAGE_WIDTH;
        int w = i % IMAGE_WIDTH;
        int neighbors = countNeighbors(greenImageData, h, w);
        Pixel cell = getCell(greenImageData, h, w);
        if (cell == LIVE) {
            if (neighbors < 2 || neighbors > 3) {
                redImageData[h * IMAGE_WIDTH + w] = DEAD;
            }
            else {
                redImageData[h * IMAGE_WIDTH + w] = LIVE;
            }
        }
        else {
            if (neighbors == 3) {
                redImageData[h * IMAGE_WIDTH + w] = LIVE;
            }
            else {
                redImageData[h * IMAGE_WIDTH + w] = DEAD;
            }
        }
    }
}


// this function creates a pool of threads and assigns each thread a portion of the image to update
void updateImageParallel(Pixel* greenImageData, Pixel* redImageData) {
    int pixelsPerThread = IMAGE_WIDTH * IMAGE_HEIGHT / NUM_THREADS;
    int start = 0;
    int end = pixelsPerThread;
    std::thread threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread(generateNextGenerationPartial, greenImageData, redImageData, start, end);
        start = end;
        end = start + pixelsPerThread;
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    return;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "OpenGL Image Processing", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Create OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);


    // Dynamically allocate the image array as a continuous block of memory
    Pixel* imageDataA = new Pixel[IMAGE_WIDTH * IMAGE_HEIGHT];
    Pixel* imageDataB = new Pixel[IMAGE_WIDTH * IMAGE_HEIGHT];
    // fill the memory
    memset(imageDataA, LIVE, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel));
    memset(imageDataB, DEAD, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel));

    // randomizeImage(imageDataA);

    fillHorizontalLines(imageDataA);
    fillVerticalLines(imageDataB);    

    // Set texture parameters and upload image data here (use glTexImage2D)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageDataA);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // Main loop
    while (!glfwWindowShouldClose(window)) {

		// Update the image data
        updateImageParallel(imageDataA, imageDataB);
        std::swap(imageDataA, imageDataB);

        // Update the texture with the new image data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageDataA);

        // Render the texture
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBegin(GL_QUADS);
        // Define vertices to render the texture onto, render the texture fullscreen
        // Coordintaes are based on the screen size
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(-1.0f, 1.0f);

        glEnd();
        glDisable(GL_TEXTURE_2D);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR)
        {
            // print the error
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
