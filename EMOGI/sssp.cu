/* References:
 *
 *    Hong, Sungpack, et al.
 *    "Accelerating CUDA graph algorithms at maximum warp."
 *    Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 *    Lifeng Nai, Yinglong Xia, Ilie G. Tanase, Hyesoon Kim, and Ching-Yung Lin.
 *    GraphBIG: Understanding Graph Computing in the Context of Industrial Solutions,
 *    In the proccedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC),
 *    Nov. 2015
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
#include <algorithm>


#define MEM_ALIGN MEM_ALIGN_32

typedef uint32_t EdgeT;
typedef uint32_t WeightT;

struct Edge{
    EdgeT src;
    EdgeT dst;
};

struct EdgeWeighted{
    EdgeT src;
    EdgeT dst;
    WeightT weight;
};


__global__ void kernel_coalesce(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && label[warpIdx]) {
        uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        uint64_t end = vertexList[warpIdx+1];

        WeightT cost = newCostList[warpIdx];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            //if (newCostList[warpIdx] != cost)
                //break;
            if (newCostList[edgeList[i]] > cost + weightList[i] && i >= start)
                atomicMin(&(newCostList[edgeList[i]]), cost + weightList[i]);
        }

        label[warpIdx] = false;
    }
}

__global__ void kernel_coalesce_chunk(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;//
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if (label[i]) {
            uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = vertexList[i+1];

            WeightT cost = newCostList[i];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[i] != cost)
                    break;
                if (newCostList[edgeList[j]] > cost + weightList[j] && j >= start)
                    atomicMin(&(newCostList[edgeList[j]]), cost + weightList[j]);
            }

            label[i] = false;
        }
    }
}

__global__ void update(bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed) {
	uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;


    if (tid < vertex_count) {
        if (newCostList[tid] < costList[tid]) {
            costList[tid] = newCostList[tid];
            label[tid] = true;
            *changed = true;
        }
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
    std::ifstream file, file2;
    std::string vertex_file, edge_file, weight_file;
    char* filename;

    bool changed_h, *changed_d, no_src = false, *label_d;
    int c, num_run = 1, arg_num = 0, device = 0;
    impl_type type;
    mem_type mem;
    uint32_t one, iter;
    WeightT offset = 0;
    WeightT zero;
    WeightT *costList_d, *newCostList_d, *weightList_h, *weightList_d;
    uint64_t *vertexList_h;
    uint64_t *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t vertex_count, edge_count, weight_count, vertex_size, edge_size, weight_size;
    uint64_t typeT, src;
    uint64_t numblocks_kernel, numblocks_update, numthreads;

    float milliseconds;
    double avg_milliseconds;
    float iter_milliseconds;
    float tran_milliseconds;
    float total_tran_milliseconds = 0.0;

    cudaEvent_t start, end;
    cudaEvent_t start_iter, end_iter;
    cudaEvent_t start_transfer,end_transfer;


    while ((c = getopt(argc, argv, "f:r:t:i:m:d:o:h")) != -1) {
        switch (c) {
            case 'f':
                filename = optarg;
                arg_num++;
                break;
            case 'r':
                if (!no_src)
                    src = atoll(optarg);
                arg_num++;
                break;
            case 't':
                type = (impl_type)atoi(optarg);
                arg_num++;
                break;
            case 'i':
                no_src = true;
                src = 0;
                num_run = atoi(optarg);
                arg_num++;
                break;
            case 'm':
                mem = (mem_type)atoi(optarg);
                arg_num++;
                break;
            case 'd':
                device = atoi(optarg);
                break;
            case 'o':
                offset = atoi(optarg);
                break;
            case 'h':
                printf("8-byte edge SSSP with uint32 edge weight\n");
                printf("\t-f | input file name (must end with .bel)\n");
                printf("\t-r | SSSP root (unused when i > 1)\n");
                printf("\t-t | type of SSSP to run\n");
                printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
                printf("\t-m | memory allocation\n");
                printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
                printf("\t-i | number of iterations to run\n");
                printf("\t-d | GPU device id (default=0)\n");
                printf("\t-o | edge weight offset (default=0)\n");
                printf("\t-h | help message\n");
                return 0;
            case '?':
                break;
            default:
                break;
        }
    }

    if (arg_num < 4) {
        printf("8-byte edge SSSP with uint32 edge weight\n");
        printf("\t-f | input file name (must end with .bel)\n");
        printf("\t-r | SSSP root (unused when i > 1)\n");
        printf("\t-t | type of SSSP to run\n");
        printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
        printf("\t-m | memory allocation\n");
        printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
        printf("\t-i | number of iterations to run\n");
        printf("\t-d | GPU device id (default=0)\n");
        printf("\t-o | edge weight offset (default=0)\n");
        printf("\t-h | help message\n");
        return 0;
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventCreate(&start_iter));
    checkCudaErrors(cudaEventCreate(&end_iter));
    checkCudaErrors(cudaEventCreate(&start_transfer));
    checkCudaErrors(cudaEventCreate(&end_transfer));

    uint64_t *xadj;
    EdgeT *vwgt, *vsize;
    WeightT *adjwgt;
    EdgeT *adjncy;
    edge_count = 0;
    vertex_count = 0;

    std::ifstream infile;
    infile.open(filename);
    std::stringstream ss;
    std::string line;
    
    printf("------------------------sssp-------------------------------\n");
    int weighted = 1;

    std::vector<EdgeT> xadj_pri;
    xadj_pri.resize(1);

    EdgeT source,dst;
    while(getline( infile, line )){


        ss.str("");
        ss.clear();
        ss << line;
                
        ss >> source;
        ss >> dst;

        if(vertex_count < source)
            vertex_count = source;
        if(vertex_count < dst)
            vertex_count = dst;

        edge_count += 2;
        if(xadj_pri.size() <= vertex_count)
        {
            xadj_pri.resize(vertex_count * 2);
        }
        xadj_pri[source]++;
        xadj_pri[dst]++;

    }
    infile.close();
    vertex_count++;

    bool *label_h = (bool *) calloc((vertex_count), sizeof(bool));
    bool *label_temp = (bool *) calloc((vertex_count*2), sizeof(bool));
    printf("%s has %lu nodes and %lu edges\n", filename, vertex_count, edge_count);

    /*uint32_t *out_in = (uint32_t *) calloc((vertex_count + 1), sizeof(uint32_t));
    uint32_t *new_id  = (uint32_t *) calloc((vertex_count + 1), sizeof(uint32_t));

    for(uint32_t source = 0; source < vertex_count; source++){

        out_in[source] = xadj_pri[source];
    }

    std::pair<uint32_t, uint32_t> seg_res_idx;
    std::vector<std::pair<uint32_t, uint32_t>> seg_res_rank;   

    
    for(uint32_t seg_idx = 0; seg_idx < vertex_count ; seg_idx++){
        seg_res_idx.first = seg_idx;    
        seg_res_idx.second = out_in[seg_idx];
        seg_res_rank.push_back(seg_res_idx);
    }
    std::sort(seg_res_rank.begin(), seg_res_rank.end(), [](std::pair<uint32_t, uint32_t> v1, std::pair<uint32_t, uint32_t> v2){
        return v1.second > v2.second;
    }); 
        
    for (uint32_t i = 0; i < vertex_count; i++)
    {
        source = seg_res_rank[i].first;
        if(source == 0)
            printf("source = 0 ,new_id: %d\n",i);
        new_id[source] = i;
    }*/

    xadj = (uint64_t *) calloc((vertex_count + 1), sizeof(uint64_t));
    adjncy = (EdgeT *) calloc((edge_count), sizeof(EdgeT));
    adjwgt = (WeightT *) calloc((edge_count), sizeof(WeightT));
    

    uint32_t edge_idx = 0;

    uint64_t count = 0;
    for (EdgeT source = 0; source < vertex_count; source++)
    {
        xadj[source] = count;
        count += xadj_pri[source];
    }
    
    xadj[vertex_count] = edge_count;

   /* uint32_t *xadj_32;
    xadj_32 = (uint32_t *) calloc((vertex_count + 1), sizeof(uint32_t));
    for(int i = 0; i < vertex_count + 1;i++){
        xadj_32[i] = xadj[i];
    }*/
   /* std::ofstream ofs("sssp_csr.idx", std::ofstream::binary|std::ofstream::app);
    ofs.write(reinterpret_cast<const char*>(xadj_32), sizeof(uint32_t) * (vertex_count+1));
    ofs.close();*/


    infile.open(filename);

    uint32_t *outDegreeCounter  = (uint32_t *) calloc((vertex_count), sizeof(uint32_t));
    for(uint32_t i=0; i<vertex_count; i++)
        outDegreeCounter[i] = 0;

    while(getline( infile, line )){
        ss.str("");
        ss.clear();
        ss << line;
                
        ss >> source;
        ss >> dst;    

 
        uint64_t location = xadj[source] + outDegreeCounter[source];                
        adjncy[location] = dst;
        outDegreeCounter[source]++; 
        adjwgt[location] = source % 64;

        location = xadj[dst] + outDegreeCounter[dst];                
        adjncy[location] = source;
        outDegreeCounter[dst]++; 
        adjwgt[location] = dst % 64;

    }

   /* std::ofstream ofs_3("sssp_csr.deg", std::ofstream::binary|std::ofstream::app);
    ofs_3.write(reinterpret_cast<const char*>(outDegreeCounter), sizeof(uint32_t) * (vertex_count));
    ofs_3.close();

    std::ofstream ofs_1("sssp_csr.ngh", std::ofstream::binary|std::ofstream::app);
    ofs_1.write(reinterpret_cast<const char*>(adjncy), sizeof(uint32_t) * (edge_count));
    ofs_1.close();

    std::ofstream ofs_2("sssp_csr.wgh", std::ofstream::binary|std::ofstream::app);
    ofs_2.write(reinterpret_cast<const char*>(adjwgt), sizeof(uint32_t) * (edge_count));
    ofs_2.close();
    
    infile.close();*/
    printf("Vertex: %lu, ", vertex_count);
    vertex_size = (vertex_count+1) * sizeof(uint64_t);

    printf("Edge: %lu\n", edge_count);

    edge_size = edge_count * sizeof(EdgeT);

    weight_count = edge_count;
    printf("Weight: %lu\n", weight_count);
    
    weight_size = weight_count * sizeof(WeightT);


    switch (mem) {
        case GPUMEM:

            checkCudaErrors(cudaMalloc((void**)&edgeList_d, edge_size));
            checkCudaErrors(cudaMalloc((void**)&weightList_d, weight_size));

            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
  	    for (uint64_t i = 0; i < edge_count; i++){
               edgeList_d[i] = (uint32_t)adjncy[i];
               weightList_d[i] = (uint32_t)adjwgt[i];
            }
            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:
            cudaHostRegister((void *)adjncy, edge_size , cudaHostRegisterMapped);
            cudaHostGetDevicePointer((void **)&edgeList_d, (void *)adjncy, 0);
            cudaHostRegister((void *)adjwgt, weight_size , cudaHostRegisterMapped);
            cudaHostGetDevicePointer((void **)&weightList_d, (void *)adjwgt, 0);
            break;
    }

    file.close();
    file2.close();

    // Allocate memory for GPU
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&label_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&costList_d, vertex_count * sizeof(WeightT)));
    checkCudaErrors(cudaMalloc((void**)&newCostList_d, vertex_count * sizeof(WeightT)));

    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    checkCudaErrors(cudaMemcpy(vertexList_d, xadj, vertex_size, cudaMemcpyHostToDevice));

    if (mem == GPUMEM) {
        checkCudaErrors(cudaMemcpy(edgeList_d, adjncy, edge_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(weightList_d, adjwgt, weight_size, cudaMemcpyHostToDevice));
    }

    numthreads = BLOCK_SIZE;

    switch (type) {
        case COALESCE:
            numblocks_kernel = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
            break;
        case COALESCE_CHUNK:
            numblocks_kernel = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }

    numblocks_update = ((vertex_count + numthreads) / numthreads);

    dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;
    iter_milliseconds = 0.0f;

    printf("Initialization done\n");
    fflush(stdout);

    uint64_t access_8 = 0;
    uint64_t access_16 = 0;
    uint64_t access_24 = 0;
    uint64_t access_32 = 0;
    // Set root
    for (int i = 0; i < num_run; i++) {
        zero = 0;
        one = 1;
        checkCudaErrors(cudaMemset(costList_d, 0xFF, vertex_count * sizeof(WeightT)));
        checkCudaErrors(cudaMemset(newCostList_d, 0xFF, vertex_count * sizeof(WeightT)));
        checkCudaErrors(cudaMemset(label_d, 0x0, vertex_count * sizeof(bool)));
        checkCudaErrors(cudaMemcpy(&label_d[src], &one, sizeof(bool), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&costList_d[src], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&newCostList_d[src], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));

        iter = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        // Run SSSP
        do {
            //checkCudaErrors(cudaEventRecord(start_iter, 0));
            changed_h = false;
            checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

            //checkCudaErrors(cudaEventRecord(start_transfer, 0));
            //if (mem == GPUMEM) {
            
            //checkCudaErrors(cudaMemcpy(edgeList_d, adjncy, edge_size, cudaMemcpyHostToDevice));
            //cudaDeviceSynchronize();
            //checkCudaErrors(cudaMemcpy(weightList_d, adjwgt, weight_size, cudaMemcpyHostToDevice));

            //}

            //checkCudaErrors(cudaEventRecord(end_transfer, 0));
            //checkCudaErrors(cudaEventSynchronize(end_transfer));
            //checkCudaErrors(cudaEventElapsedTime(&tran_milliseconds, start_transfer, end_transfer));
            //printf("time %*f ms ", 12, tran_milliseconds);

            switch (type) {
                case COALESCE:
                    kernel_coalesce<<<blockDim_kernel, numthreads>>>(label_d, costList_d, newCostList_d, vertex_count, vertexList_d, edgeList_d, weightList_d);
                    break;
                case COALESCE_CHUNK:
                    kernel_coalesce_chunk<<<blockDim_kernel, numthreads>>>(label_d, costList_d, newCostList_d, vertex_count, vertexList_d, edgeList_d, weightList_d);
                    break;
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }
            cudaDeviceSynchronize();
            update<<<blockDim_update, numthreads>>>(label_d, costList_d, newCostList_d, vertex_count, changed_d);

            iter++;
            //checkCudaErrors(cudaEventRecord(end_iter, 0));
            //checkCudaErrors(cudaEventSynchronize(end_iter));
            //checkCudaErrors(cudaEventElapsedTime(&iter_milliseconds, start_iter, end_iter));
            //printf("time %*f ms", 12, iter_milliseconds);

            checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(label_h, label_d, sizeof(bool) * vertex_count, cudaMemcpyDeviceToHost));

        } while(changed_h);


        checkCudaErrors(cudaEventRecord(end, 0));
        checkCudaErrors(cudaEventSynchronize(end));
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));

        printf("run %*d: ", 3, i);
        printf("src %*lu, ", 12, src);
        printf("iteration %*u, ", 3, iter);
        printf("time %*f ms\n", 12, (milliseconds - total_tran_milliseconds));
        fflush(stdout);

        avg_milliseconds += (double)milliseconds;

        src += vertex_count / num_run;

        if (i < num_run - 1) {
            EdgeT *edgeList_temp;
            WeightT *weightList_temp;

            // Flush GPU page cache for each iteration by re-allocating UVM
            switch (mem) {
                case UVM_READONLY:
                    checkCudaErrors(cudaMallocManaged((void**)&edgeList_temp, edge_size));
                    checkCudaErrors(cudaMallocManaged((void**)&weightList_temp, weight_size));
                    memcpy(edgeList_temp, edgeList_d, edge_size);
                    memcpy(weightList_temp, weightList_d, weight_size);
                    checkCudaErrors(cudaFree(edgeList_d));
                    checkCudaErrors(cudaFree(weightList_d));
                    edgeList_d = edgeList_temp;
                    weightList_d = weightList_temp;
                    checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
                    checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, device));
                    break;
                default:
                    break;
            }
        }
    }
    printf("num_run:%d",num_run);
    printf("Average run time %f ms\n", avg_milliseconds / num_run);

    free(vertexList_h);
    if (edgeList_h)
        free(edgeList_h);
    if (weightList_h)
        free(weightList_h);
    checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(weightList_d));
    checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(costList_d));
    checkCudaErrors(cudaFree(newCostList_d));
    checkCudaErrors(cudaFree(label_d));
    checkCudaErrors(cudaFree(changed_d));

    return 0;
}
