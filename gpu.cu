#include "common.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define NUM_THREADS 256

// Grid & Simulation parameters
static int grid_dim = 0;
static int total_cells = 0;
static double cell_size = 0.0;
static int part_blocks = 0;
static int current_step = 0;

// Device buffers
static int* d_cell_ids = nullptr;
static int* d_particle_indices = nullptr;
static int* d_orig_id = nullptr;
static int* d_sorted_orig_id = nullptr;
static particle_t* d_sorted_parts = nullptr;
static int* d_cell_start = nullptr;
static int* d_cell_end = nullptr;

__device__ void apply_force_gpu(particle_t& particle, const particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    double coef = (1.0 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void init_indices_gpu(int* orig_id, int num_parts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_parts) {
        orig_id[tid] = tid;
    }
}

__global__ void compute_cell_ids_gpu(const particle_t* parts, int* cell_ids, int* indices, 
                                     int num_parts, int g_dim, double c_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int cx = (int)(parts[tid].x / c_size);
    int cy = (int)(parts[tid].y / c_size);

    cx = min(max(cx, 0), g_dim - 1);
    cy = min(max(cy, 0), g_dim - 1);

    cell_ids[tid] = cy * g_dim + cx;
    indices[tid] = tid;
}

__global__ void reorder_particles_and_ids_gpu(particle_t* sorted_parts, const particle_t* parts,
                                              int* sorted_orig_id, const int* orig_id,
                                              const int* indices, int num_parts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int orig_buf_idx = indices[tid];
    sorted_parts[tid] = parts[orig_buf_idx];
    sorted_orig_id[tid] = orig_id[orig_buf_idx];
}

__global__ void find_cell_bounds_gpu(const int* cell_ids, int* cell_start, int* cell_end, 
                                     int num_parts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int cell = cell_ids[tid];

    if (tid == 0 || cell != cell_ids[tid - 1]) {
        cell_start[cell] = tid;
    }
    if (tid == num_parts - 1 || cell != cell_ids[tid + 1]) {
        cell_end[cell] = tid + 1;
    }
}

__global__ void compute_forces_grid_gpu(particle_t* sorted_parts, const int* cell_start, 
                                        const int* cell_end, int num_parts, 
                                        int g_dim, double c_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t p = sorted_parts[tid];
    p.ax = 0.0;
    p.ay = 0.0;

    int cx = (int)(p.x / c_size);
    int cy = (int)(p.y / c_size);
    cx = min(max(cx, 0), g_dim - 1);
    cy = min(max(cy, 0), g_dim - 1);

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = cy + dy;
        if (ny < 0 || ny >= g_dim) continue;

        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= g_dim) continue;

            int n_cell = ny * g_dim + nx;
            int start = cell_start[n_cell];
            if (start == -1) continue;

            int end = cell_end[n_cell];
            for (int j = start; j < end; ++j) {
                apply_force_gpu(p, sorted_parts[j]);
            }
        }
    }

    sorted_parts[tid].ax = p.ax;
    sorted_parts[tid].ay = p.ay;
}

__global__ void move_sorted_gpu(particle_t* sorted_parts, int num_parts, double size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t* p = &sorted_parts[tid];

    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    while (p->x < 0.0 || p->x > size) {
        p->x = (p->x < 0.0) ? -(p->x) : 2.0 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0.0 || p->y > size) {
        p->y = (p->y < 0.0) ? -(p->y) : 2.0 * size - p->y;
        p->vy = -(p->vy);
    }
}

__global__ void restore_particles_gpu(particle_t* dest, const particle_t* src,
                                      const int* orig_indices, int num_parts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int orig_idx = orig_indices[tid];
    dest[orig_idx] = src[tid];
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    part_blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    current_step = 0;

    grid_dim = (int)(size / cutoff);
    if (grid_dim < 1) grid_dim = 1;
    cell_size = size / grid_dim;
    total_cells = grid_dim * grid_dim;

    cudaMalloc((void**)&d_cell_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&d_particle_indices, num_parts * sizeof(int));
    cudaMalloc((void**)&d_orig_id, num_parts * sizeof(int));
    cudaMalloc((void**)&d_sorted_orig_id, num_parts * sizeof(int));
    cudaMalloc((void**)&d_sorted_parts, num_parts * sizeof(particle_t));
    cudaMalloc((void**)&d_cell_start, total_cells * sizeof(int));
    cudaMalloc((void**)&d_cell_end, total_cells * sizeof(int));

    init_indices_gpu<<<part_blocks, NUM_THREADS>>>(d_orig_id, num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    cudaMemset(d_cell_start, -1, total_cells * sizeof(int));
    cudaMemset(d_cell_end, -1, total_cells * sizeof(int));

    compute_cell_ids_gpu<<<part_blocks, NUM_THREADS>>>(
        parts, d_cell_ids, d_particle_indices, num_parts, grid_dim, cell_size);

    thrust::device_ptr<int> dev_keys(d_cell_ids);
    thrust::device_ptr<int> dev_vals(d_particle_indices);
    thrust::sort_by_key(dev_keys, dev_keys + num_parts, dev_vals);

    reorder_particles_and_ids_gpu<<<part_blocks, NUM_THREADS>>>(
        d_sorted_parts, parts, d_sorted_orig_id, d_orig_id, d_particle_indices, num_parts);

    find_cell_bounds_gpu<<<part_blocks, NUM_THREADS>>>(
        d_cell_ids, d_cell_start, d_cell_end, num_parts);

    compute_forces_grid_gpu<<<part_blocks, NUM_THREADS>>>(
        d_sorted_parts, d_cell_start, d_cell_end, num_parts, grid_dim, cell_size);

    move_sorted_gpu<<<part_blocks, NUM_THREADS>>>(
        d_sorted_parts, num_parts, size);

    if ((current_step % savefreq) == 0) {
        restore_particles_gpu<<<part_blocks, NUM_THREADS>>>(
            parts, d_sorted_parts, d_sorted_orig_id, num_parts);

        init_indices_gpu<<<part_blocks, NUM_THREADS>>>(d_orig_id, num_parts);
    } else {
        cudaMemcpy(parts, d_sorted_parts, num_parts * sizeof(particle_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_orig_id, d_sorted_orig_id, num_parts * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    current_step++;
}

void clear_simulation() {
    if (d_cell_ids) cudaFree(d_cell_ids);
    if (d_particle_indices) cudaFree(d_particle_indices);
    if (d_orig_id) cudaFree(d_orig_id);
    if (d_sorted_orig_id) cudaFree(d_sorted_orig_id);
    if (d_sorted_parts) cudaFree(d_sorted_parts);
    if (d_cell_start) cudaFree(d_cell_start);
    if (d_cell_end) cudaFree(d_cell_end);

    d_cell_ids = nullptr;
    d_particle_indices = nullptr;
    d_orig_id = nullptr;
    d_sorted_orig_id = nullptr;
    d_sorted_parts = nullptr;
    d_cell_start = nullptr;
    d_cell_end = nullptr;
}