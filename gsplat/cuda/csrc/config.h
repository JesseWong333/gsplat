#define BLOCK_X 1
#define BLOCK_Y 1
#define BLOCK_Z 1
// #define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define N_THREADS 256

#define MAX_REGISTER_CHANNELS 256
#define MAX_POINTS_PER_THREAD 16  // 一个 block最多处理 256 * 16 = 4096 个点
#define CHANNELS 2

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf(                                                            \
                "Error at %s:%d - %s\n",                                       \
                __FILE__,                                                      \
                __LINE__,                                                      \
                cudaGetErrorString(cudaGetLastError())                         \
            );                                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
