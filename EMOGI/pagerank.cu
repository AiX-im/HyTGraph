/* References:
 *
 *    Hong, Sungpack, et al.
 *    "Accelerating CUDA graph algorithms at maximum warp."
 *    Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 *    There are so many PageRank algorithms available. We use something similar to:
 *        Galois: https://github.com/IntelligentSoftwareSystems/Galois/blob/master/lonestar/analytics/cpu/pagerank/PageRank-push.cpp
 *
 */

#include "helper_emogi.h"
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#define MEM_ALIGN MEM_ALIGN_64
#define SCIDX  "lu"

typedef uint32_t EdgeT;
typedef float ValueT;

struct Edge{
    EdgeT src;
    EdgeT dst;
};

struct EdgeWeighted{
    EdgeT src;
    EdgeT dst;
    EdgeT weight;
};

__global__ void initialize(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
	}
}

__global__ void kernel_coalesce(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start)
                atomicAdd(&residual[edgeList[i]], delta[warpIdx]);

        label[warpIdx] = false;
    }
}
__global__ void test(bool *changed){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if(tid == 0){
      if(*changed == false){  
      printf("test_test\n");
        }
    }
}
__global__ void kernel_coalesce_chunk(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed) {
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
        if(label[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE)
                if (j >= start)
                    atomicAdd(&residual[edgeList[j]], delta[i]);

            label[i] = false;
        }
    }
}

__global__ void update(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        delta[tid] = residual[tid] * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
	}
}





FILE *gk_fopen(const char *fname, const char *mode, const char *msg)
{
    FILE *fp;
    char errmsg[8192];

    fp = fopen(fname, mode);
    if (fp != NULL){
        return fp;
    }

    else{
        printf("openfail\n");
        exit(1);
    }
    return NULL;
}

ptrdiff_t gk_getline(char **lineptr, size_t *n, FILE *stream)
{
#ifdef HAVE_GETLINE
    return getline(lineptr, n, stream);
#else
    size_t i;
    int ch;

    if (feof(stream))
        return -1;

    /* Initial memory allocation if *lineptr is NULL */
    if (*lineptr == NULL || *n == 0)
    {
        *n = 1024;
        *lineptr = (char *) malloc((*n) * sizeof(char));
    }

    /* get into the main loop */
    i = 0;
    while ((ch = getc(stream)) != EOF)
    {
        (*lineptr)[i++] = (char) ch;

        /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
        if (i + 1 == *n)
        {
            *n = 2 * (*n);
            *lineptr = (char *) realloc(*lineptr, (*n) * sizeof(char));
        }

        if (ch == '\n')
            break;
    }
    (*lineptr)[i] = '\0';

    return (i == 0 ? -1 : i);
#endif
}


int main(int argc, char *argv[]) {
    std::ifstream file;
    std::string vertex_file, edge_file;
    char* filename;

    bool changed_h, *changed_d, *label_d;
    int c, arg_num = 0, device = 0;
    impl_type type;
    mem_type mem;
    ValueT *delta_d, *residual_d, *value_d, *value_h;
    ValueT tolerance, alpha;
    uint32_t iter, max_iter;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t numblocks, numblocks_update, numthreads;
    uint64_t typeT;

    float milliseconds;
    float iter_milliseconds;
    float tran_milliseconds;
    double avg_milliseconds;
    float total_tran_milliseconds = 0.0;

    cudaEvent_t start, end;
    cudaEvent_t start_iter, end_iter;
    cudaEvent_t start_transfer,end_transfer;

    alpha = 0.85;
    tolerance = 0.01;
    max_iter = 5000;

    while ((c = getopt(argc, argv, "f:t:m:d:a:l:i:h")) != -1) {
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
            case 'a':
                alpha = atof(optarg);
                break;
            case 'l':
                tolerance = atof(optarg);
                break;
            case 'i':
                max_iter = atoi(optarg);
                break;
            case 'h':
                printf("8-byte edge PageRank\n");
                printf("\t-f | input file name (must end with .bel)\n");
                printf("\t-t | type of PageRank to run\n");
                printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
                printf("\t-m | memory allocation\n");
                printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
                printf("\t-d | GPU device id (default=0)\n");
                printf("\t-a | alpha (default=0.85)\n");
                printf("\t-l | tolerance (default=0.001)\n");
                printf("\t-i | max iteration (default=5000)\n");
                printf("\t-h | help message\n");
                return 0;
            case '?':
                break;
            default:
                break;
        }
    }

    if (arg_num < 3) {
        printf("8-byte edge PageRank\n");
        printf("\t-f | input file name (must end with .bel)\n");
        printf("\t-t | type of PageRank to run\n");
        printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
        printf("\t-m | memory allocation\n");
        printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
        printf("\t-d | GPU device id (default=0)\n");
        printf("\t-a | alpha (default=0.85)\n");
        printf("\t-l | tolerance (default=0.001)\n");
        printf("\t-i | max iteration (default=5000)\n");
        printf("\t-h | help message\n");
        return 0;
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventCreate(&start_iter));
    checkCudaErrors(cudaEventCreate(&end_iter));
    checkCudaErrors(cudaEventCreate(&start_transfer));
    checkCudaErrors(cudaEventCreate(&end_transfer));

    uint64_t *xadj, *vwgt, *vsize;
    EdgeT *adjwgt, *adjncy;

    edge_count = 0;
    vertex_count = 0;
    printf("-------------------------------------pagerank----------------------------------\n");
    std::ifstream infile;
    infile.open(filename);
    std::stringstream ss;
    std::string line;

    bool first_line = true;
    int weighted = 0;

    std::vector<EdgeT> xadj_pri;
    std::vector<EdgeT> indegree;
    xadj_pri.resize(1);
    indegree.resize(1);

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
        //printf("edge:%d\n",edge_count);  
        if(xadj_pri.size() <= vertex_count)
        {
            xadj_pri.resize(vertex_count * 2);
            //indegree.resize(vertex_count * 2);
        }
        xadj_pri[src]++;
        xadj_pri[dst]++;
        //indegree[dst]++;

    }
    infile.close();
    vertex_count++;


    //bool label_h[vertex_count];
    bool *label_h = (bool *) calloc((vertex_count), sizeof(bool));
    //uint32_t *in_out = (uint32_t *) calloc((vertex_count + 1), sizeof(uint32_t));
    //uint32_t *new_id  = (uint32_t *) calloc((vertex_count + 1), sizeof(uint32_t));

    /*for(uint32_t source = 0; source < vertex_count; source++){
        in_out[source] = indegree[source];
    }

    std::pair<uint32_t, uint32_t> seg_res_idx_in;
    std::vector<std::pair<uint32_t, uint32_t>> seg_res_rank_in;  

    
    for(uint32_t seg_idx = 0; seg_idx < vertex_count ; seg_idx++){
        seg_res_idx_in.first = seg_idx;
        seg_res_idx_in.second = in_out[seg_idx];
        seg_res_rank_in.push_back(seg_res_idx_in);
    }

    std::sort(seg_res_rank_in.begin(), seg_res_rank_in.end(), [](std::pair<uint32_t, uint32_t> v1, std::pair<uint32_t, uint32_t> v2){
        return v1.second > v2.second;
    }); 
       

    for (uint32_t i = 0; i < vertex_count; i++)
    {
        src = seg_res_rank_in[i].first;
        if(src == 0)
            printf("source = %d ,new_id: %d\n",src,i);
        new_id[src] = i;
    }*/


    //sscanf(line,"%" SCIDX " %" SCIDX " %" SCIDX, &(vertex_count), &(vertex_count), &(edge_count));
    printf("%s has %lu nodes and %lu edges\n", filename, vertex_count, edge_count);

    //return 0;
    xadj = (uint64_t *) calloc((vertex_count + 1), sizeof(uint64_t));

    adjncy = (EdgeT *) calloc((edge_count), sizeof(EdgeT));
 

    uint32_t edge_idx = 0;

    uint64_t count = 0;
    for (uint32_t src = 0; src < vertex_count; src++)
    {
        xadj[src] = count;
        count += xadj_pri[src];
    }
    xadj[vertex_count] = edge_count;

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


    /*std::ofstream outfile("out_in.el",std::ofstream::app);

    

    for (uint32_t i = 0; i < vertex_count; i++)
    {
        src = seg_res_rank_in[i].first;

        new_id[src] = i;
    }

    for (uint32_t i = 0; i < vertex_count; i++)
    {
        src = seg_res_rank_in[i].first;
        uint64_t start = xadj[src];
        uint64_t end = xadj[src + 1];

        for(uint32_t j = start; j < end; j++){
            outfile << i << '\t' << new_id[adjncy[j]] << '\n';
        }
    }
    outfile.close();
    return 0;*/

    printf("Vertex: %lu, ", vertex_count);
    vertex_size = (vertex_count+1) * sizeof(uint64_t);

    printf("Edge: %lu\n", edge_count);

    edge_size = edge_count * sizeof(EdgeT);


    // Allocate memory for GPU
    checkCudaErrors(cudaMalloc((void**)&label_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&delta_d, vertex_count * sizeof(ValueT)));
    checkCudaErrors(cudaMalloc((void**)&residual_d, vertex_count * sizeof(ValueT)));
    checkCudaErrors(cudaMalloc((void**)&value_d, vertex_count * sizeof(ValueT)));

    value_h = (ValueT*)malloc(vertex_count * sizeof(ValueT));

    switch (mem) {
        case GPUMEM:
            checkCudaErrors(cudaMalloc((void**)&edgeList_d, edge_size));

            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            //edgeList_d = adjncy;
	    for (uint64_t i = 0; i < edge_count; i++)
               edgeList_d[i] = (uint32_t)adjncy[i];
            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:
            //checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            //edgeList_d = adjncy;
            //checkCudaErrors(cudaMemAdvise(edgeList_d,edge_size, cudaMemAdviseSetAccessedBy, device));
            cudaHostRegister((void *)adjncy, edge_size , cudaHostRegisterMapped);
            cudaHostGetDevicePointer((void **)&edgeList_d, (void *)adjncy, 0);

            break;
    }


    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    checkCudaErrors(cudaMemcpy(vertexList_d, xadj, vertex_size, cudaMemcpyHostToDevice));

    if (mem == GPUMEM)
        checkCudaErrors(cudaMemcpy(edgeList_d, adjncy, edge_size, cudaMemcpyHostToDevice));

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

    numblocks_update = ((vertex_count + numthreads) / numthreads);

    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    iter = 0;

    printf("Initialization done\n");
    fflush(stdout);

    checkCudaErrors(cudaEventRecord(start, 0));

    initialize<<<blockDim_update, numthreads>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, alpha);
    cudaThreadSynchronize();



    // Run PageRank
    do {
        checkCudaErrors(cudaEventRecord(start_iter, 0));
        checkCudaErrors(cudaEventRecord(start_transfer, 0));
        if (mem == GPUMEM)
            checkCudaErrors(cudaMemcpy(edgeList_d, adjncy, edge_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaEventRecord(end_transfer, 0));
        checkCudaErrors(cudaEventSynchronize(end_transfer));
        checkCudaErrors(cudaEventElapsedTime(&tran_milliseconds, start_transfer, end_transfer));
        //printf("time %*f ms ", 12, tran_milliseconds);

        changed_h = false;
        //printf("iter:%d\n",iter);
        checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));
        //test<<<blockDim_update, numthreads>>>(changed_d);
        switch (type) {
            case COALESCE:
                kernel_coalesce<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                cudaThreadSynchronize();
                break;
            case COALESCE_CHUNK:
                kernel_coalesce_chunk<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d,changed_d);
                cudaThreadSynchronize();
                break;
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }
        //test<<<blockDim_update, numthreads>>>(changed_d);
        update<<<blockDim_update, numthreads>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, tolerance, alpha, changed_d);
        cudaThreadSynchronize();
        
       checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));

        iter++;
            checkCudaErrors(cudaEventRecord(end_iter, 0));
            checkCudaErrors(cudaEventSynchronize(end_iter));
            checkCudaErrors(cudaEventElapsedTime(&iter_milliseconds, start_iter, end_iter));
            //printf("time %*f ms", 12, iter_milliseconds);
            
            checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(label_h, label_d, sizeof(bool) * vertex_count, cudaMemcpyDeviceToHost));
    }while(changed_h);
    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));

    printf("iteration %*u, ", 3, iter);
    printf("time %*f ms\n", 12, (milliseconds - total_tran_milliseconds));
    fflush(stdout);

    avg_milliseconds += (double)milliseconds;

    //checkCudaErrors(cudaMemcpy(value_h, value_d, vertex_count * sizeof(ValueT), cudaMemcpyDeviceToHost));

    free(value_h);
    checkCudaErrors(cudaFree(label_d));
    checkCudaErrors(cudaFree(changed_d));
    checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(delta_d));
    checkCudaErrors(cudaFree(residual_d));
    checkCudaErrors(cudaFree(value_d));

    return 0;
}
