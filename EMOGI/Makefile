NVCC=nvcc
NVCCFLAGS=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70
CXXFLAGS= -std=c++11

all: bfs bfs_32 cc cc_32 sssp pagerank

bfs: bfs.cu
	${NVCC} -O3 ${CXXFLAGS} ${NVCCFLAGS} -o $@ $^

bfs_32: bfs_32.cu
	${NVCC} -O3 ${CXXFLAGS} ${NVCCFLAGS} -o $@ $^

cc: cc.cu
	${NVCC} -O3 ${CXXFLAGS} ${NVCCFLAGS} -o $@ $^

cc_32: cc_32.cu
	${NVCC} -O3 ${CXXFLAGS} ${NVCCFLAGS} -o $@ $^

sssp: sssp.cu
	${NVCC} -O3 ${CXXFLAGS} ${NVCCFLAGS} -o $@ $^

pagerank: pagerank.cu
	${NVCC} -O3 ${CXXFLAGS} ${NVCCFLAGS} -o $@ $^

clean:
	rm -f bfs bfs_32 cc cc_32 sssp sssp_float sssp_32 pagerank pagerank_32
