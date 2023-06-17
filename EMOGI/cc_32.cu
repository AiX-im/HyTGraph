/* References:
 *
 *     Hong, Sungpack, et al.
 *     "Accelerating CUDA graph algorithms at maximum warp."
 *     Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 *     Zhen Xu, Xuhao Chen, Jie Shen, Yang Zhang, Cheng Chen, Canqun Yang,
 *     GARDENIA: A Domain-specific Benchmark Suite for Next-generation Accelerators,
 *     ACM Journal on Emerging Technologies in Computing Systems, 2018.
 *
 */

#include "helper_emogi.h"
#include <set>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>

#define MEM_ALIGN MEM_ALIGN_32

typedef uint32_t EdgeT;

__global__ void kernel_coalesce(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && curr_visit[warpIdx] == true) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                unsigned long long comp_src = comp[warpIdx];
                const EdgeT next = edgeList[i];

                unsigned long long comp_next = comp[next];
                unsigned long long comp_target;
                EdgeT next_target;

                if (comp_next != comp_src) {
                    if (comp_src < comp_next) {
                        next_target = next;
                        comp_target = comp_src;
                        atomicMin(&comp[next_target], comp_target);
                        next_visit[next_target] = true;
                        *changed = true;
                    }
                    /*else {
                        next_target = warpIdx;
                        comp_target = comp_next;
                    }*/
   
                }
            }
        }
    }
}

__global__ void kernel_coalesce_chunk(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(curr_visit[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    unsigned long long comp_src = comp[i];
                    const EdgeT next = edgeList[j];

                    unsigned long long comp_next = comp[next];
                    unsigned long long comp_target;
                    EdgeT next_target;

                    if (comp_next != comp_src) {
                        if (comp_src < comp_next) {
                            next_target = next;
                            comp_target = comp_src;
                         atomicMin(&comp[next_target], comp_target);
                         next_visit[next_target] = true;
                        *changed = true;
                        }

                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    std::ifstream file;
    std::string vertex_file, edge_file;
    std::string filename;

    bool changed_h, *changed_d;
    bool *curr_visit_d, *next_visit_d, *comp_check;
    int c, arg_num = 0, device = 0;
    impl_type type;
    mem_type mem;
    uint32_t iter, comp_total = 0;
    unsigned long long *comp_d, *comp_h;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t *edgeList64_h;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t typeT;
    uint64_t numblocks, numthreads;

    float milliseconds;

    cudaEvent_t start, end;

    while ((c = getopt(argc, argv, "f:t:m:d:h")) != -1) {
        switch (c) {
            case 'f':
                filename = optarg;
                arg_num++;
                break;
            case 't':
                type = (impl_type)atoi(optarg);
                arg_num++;
                break;
            case 'm':
                mem = (mem_type)atoi(optarg);
                arg_num++;
                break;
            case 'd':
                device = atoi(optarg);
                break;
            case 'h':
                printf("4-byte edge CC, only works correctly with undirected graphs\n");
                printf("\t-f | input file name (must end with .bel)\n");
                printf("\t-t | type of BFS to run\n");
                printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
                printf("\t-m | memory allocation\n");
                printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
                printf("\t-d | GPU device id (default=0)\n");
                printf("\t-h | help message\n");
                return 0;
            case '?':
                break;
            default:
                break;
        }
    }

    if (arg_num < 3) {
        printf("4-byte edge CC, only works correctly with undirected graphs\n");
        printf("\t-f | input file name (must end with .bel)\n");
        printf("\t-t | type of BFS to run\n");
        printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
        printf("\t-m | memory allocation\n");
        printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
        printf("\t-d | GPU device id (default=0)\n");
        printf("\t-h | help message\n");
        return 0;
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    uint64_t *xadj, *vwgt, *vsize;
    EdgeT *adjwgt, *adjncy;

    edge_count = 0;
    vertex_count = 0;
    printf("-------------------------------------CC----------------------------------\n");
    std::ifstream infile;
    infile.open(filename);
    std::stringstream ss;
    std::string line;

    bool first_line = true;
    int weighted = 0;

    std::vector<EdgeT> xadj_pri;
    xadj_pri.resize(1);

    EdgeT src;
    EdgeT dst;
    while(getline( infile, line )){


        ss.str("");
        ss.clear();
        ss << line;
                
        ss >> src;
        ss >> dst;

        if(vertex_count < src)
            vertex_count = src;
        if(vertex_count < dst)
            vertex_count = dst;

        edge_count += 2;

        if(xadj_pri.size() <= vertex_count)
        {
            xadj_pri.resize(vertex_count * 2);
        }
        xadj_pri[src]++;
        xadj_pri[dst]++;


    }
    infile.close();
    vertex_count++;

    bool label_h[vertex_count];

    //sscanf(line,"%" SCIDX " %" SCIDX " %" SCIDX, &(vertex_count), &(vertex_count), &(edge_count));
    printf("%s has %lu nodes and %lu edges\n", filename, vertex_count, edge_count);

    xadj = (uint64_t *) calloc((vertex_count + 1), sizeof(uint64_t));

    adjncy = (EdgeT *) calloc((edge_count), sizeof(EdgeT));
 

    uint32_t edge_idx = 0;

    uint64_t count = 0;
    for (uint64_t src = 0; src < vertex_count; src++)
    {
        xadj[src] = count;
        count += xadj_pri[src];
    }
    xadj[vertex_count] = vertex_count;

    infile.open(filename);

    uint32_t *outDegreeCounter  = (uint32_t *) calloc((vertex_count + 1), sizeof(uint32_t));
    for(uint32_t i=0; i<vertex_count; i++)
        outDegreeCounter[i] = 0;

    while(getline( infile, line )){

        ss.str("");
        ss.clear();
        ss << line;
                
        ss >> src;
        ss >> dst;   

        uint64_t location = xadj[src] + outDegreeCounter[src];                
        adjncy[location] = dst;
        outDegreeCounter[src]++; 
        
        location = xadj[dst] + outDegreeCounter[dst];                
        adjncy[location] = src;
        outDegreeCounter[dst]++;

    }
    infile.close();


    printf("Vertex: %lu, ", vertex_count);
    vertex_size = (vertex_count+1) * sizeof(uint64_t);

    printf("Edge: %lu\n", edge_count);

    edge_size = edge_count * sizeof(EdgeT);

    switch (mem) {
        case GPUMEM:
            edgeList_h = (EdgeT*)malloc(edge_size);
            checkCudaErrors(cudaMalloc((void**)&edgeList_d, edge_size));

            for (uint64_t i = 0; i < edge_count; i++)
                edgeList_h[i] = (uint32_t)edgeList64_h[i];

            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));

            for (uint64_t i = 0; i < edge_count; i++)
                edgeList_d[i] = (uint32_t)adjncy[i];

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:

            cudaHostRegister((void *)adjncy, edge_size , cudaHostRegisterMapped);
            cudaHostGetDevicePointer((void **)&edgeList_d, (void *)adjncy, 0);
            break;
    }

    free(edgeList64_h);
    file.close();

    // Allocate memory for GPU
    comp_h = (unsigned long long*)malloc(vertex_count * sizeof(unsigned long long));
    comp_check = (bool*)malloc(vertex_count * sizeof(bool));
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&curr_visit_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&next_visit_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&comp_d, vertex_count * sizeof(unsigned long long)));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));

    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    for (uint64_t i = 0; i < vertex_count; i++)
        comp_h[i] = i;

    memset(comp_check, 0, vertex_count * sizeof(bool));

    checkCudaErrors(cudaMemset(curr_visit_d, 0x01, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMemset(next_visit_d, 0x00, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMemcpy(comp_d, comp_h, vertex_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(vertexList_d, xadj, vertex_size, cudaMemcpyHostToDevice));

    if (mem == GPUMEM)
        checkCudaErrors(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));

    numthreads = BLOCK_SIZE;

    switch (type) {
        case COALESCE:
            numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
            break;
        case COALESCE_CHUNK:
            numblocks = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }

    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);

    printf("Initialization done\n");
    fflush(stdout);

    iter = 0;

    checkCudaErrors(cudaEventRecord(start, 0));

    // Run CC
    do {
        changed_h = false;
        checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

        switch (type) {
            case COALESCE:
                kernel_coalesce<<<blockDim, numthreads>>>(curr_visit_d, next_visit_d, vertex_count, vertexList_d, edgeList_d, comp_d, changed_d);
                break;
            case COALESCE_CHUNK:
                kernel_coalesce_chunk<<<blockDim, numthreads>>>(curr_visit_d, next_visit_d, vertex_count, vertexList_d, edgeList_d, comp_d, changed_d);
                break;
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }

        checkCudaErrors(cudaMemset(curr_visit_d, 0x00, vertex_count * sizeof(bool)));

        bool *temp = curr_visit_d;
        curr_visit_d = next_visit_d;
        next_visit_d = temp;

        iter++;

        checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(changed_h);

    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));

    checkCudaErrors(cudaMemcpy(comp_h, comp_d, vertex_count * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < vertex_count; i++) {
        if (comp_check[comp_h[i]] == false) {
            comp_check[comp_h[i]] = true;
            comp_total++;
        }
    }

    printf("total iterations: %u\n", iter);
    printf("total components: %u\n", comp_total);
    printf("total time: %f ms\n", milliseconds);
    fflush(stdout);

    free(xadj);
    free(comp_check);
    free(comp_h);
    checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(changed_d));
    checkCudaErrors(cudaFree(comp_d));
    checkCudaErrors(cudaFree(curr_visit_d));
    checkCudaErrors(cudaFree(next_visit_d));

    return 0;
}
