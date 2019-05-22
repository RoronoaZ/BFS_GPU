#include <stdio.h> 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <stdlib.h> 
#include <string.h> 
#include <string> 
#include <omp.h> 
#include <time.h> 
#include <cuda.h> 
#include <limits> 
#include <vector> 
#include <cstdlib>

//#define DEBUGC //#define TOPDOWN //#define BOTTOMUP
#define HYBRID
//#define CHECK_DEPTH_RESULT
#define ALPHA 128
using namespace std; 

int ** read_mtx(char*fname){
  const char* filename_input = fname;
  string line;
  string path = "../../data/graphs/"; //path of the graph directory
  string filename(filename_input);
  string fullPath = path + filename;
  char cstr[fullPath.size() + 1];
  strcpy(cstr, fullPath.c_str());
  ifstream input (cstr);
  if(input.fail()) //open file
    return 0;
  // int tuplesStartCounter = 0;
  //pass information part in the file
  while(getline(input, line)){
    if(line.at(0) == '%'){
      continue;
    }
    else{
      break;
    }
  }
  //get vertex and edge amount
  int vertex_amount, max_vertex_id, tuple_amount, edge_amount;
  stringstream ss(line);
  ss >> max_vertex_id;
  ss >> max_vertex_id;
  ss >> tuple_amount;
  edge_amount = 2*tuple_amount;
  vertex_amount = max_vertex_id + 1;
  #ifdef DEBUG
  cout << "VA: " << vertex_amount << " EA: " << edge_amount << endl;
  #endif
  int **M;
  M = new int*[tuple_amount];
  for(int i = 0; i < tuple_amount; i ++) //allocate adjacency matrix
    M[i] = new int[2]();
  #ifdef DEBUG
  cout << "Tuple Matrix is allocated." << endl;
  #endif
  int v, w;
  for(int i = 0; getline(input, line); i++){ //populate adjacency matrix
    stringstream ss(line);
    ss >> v;
    ss >> w;
    M[i][0] = v;
    M[i][1] = w;
  }
  #ifdef DEBUG
  cout << "Tuple Matrix is created. Ex. M[0][0] = " << M[0][0] << " M[0][1] = " << M[0][1] << endl;
  #endif
  int * V = new int[vertex_amount+1] (); // allocate array for vertices
  //V[vertex_amount] = edge_amount; // last element will point to the end of the list
  for(int i = 0; i < tuple_amount; i++){ //counting edge amount per vertex
    V[M[i][0]+1]++;
    V[M[i][1]+1]++;
  }
  #ifdef DEBUG
  cout << "Edge amount per vertex counted and stored." << endl;
  #endif
  int edgeStart = 0;
  for(int i = 0; i < vertex_amount; i++ ){ //point vertices to the correct index
    if(V[i+1] == 0){ //mark the isolated edge
      V[i] = -1;
    }
    else{
      edgeStart += V[i+1];
    }
    V[i+1] = edgeStart; //correct pointers
  }
  #ifdef DEBUG
  cout << "Edge start locations per vertex corrected and isolated edges are marked." << endl;
  #endif
  int * A = new int[edgeStart] (); // allocate array for the edges
  int * AEC = new int[vertex_amount](); // added edge count per vertex
  for(int i = 0; i < tuple_amount; i++ ){ //construct A
    A[V[M[i][0]] + AEC[M[i][0]]] = M[i][1]; //put to A
    A[V[M[i][1]] + AEC[M[i][1]]] = M[i][0]; //put to A
    AEC[M[i][0]]++; //update count
    AEC[M[i][1]]++; //update count
  }
  #ifdef DEBUG
  for(int i = max_vertex_id - 1; i < max_vertex_id + 2; i++){
    cout << "V is constructed. V[" << i << "] = " << V[i] << endl;
  }
  /*for(int i = 0; i < edge_amount; i++){
    cout << "A is constructed. A[V[" << i << "]] = " << A[V[i]] << endl;
    }*/
  #endif
  int ** ret = new int*[3];
  ret[0] = V;
  ret[1] = A;
  ret[2] = new int(max_vertex_id+1);
  ret[2][1] = edge_amount;
  return ret;
}
__global__ void BottomUp(int num_vertices, int * d_V, int * d_A, int level, int * d_depth, bool * d_still_running, int * d_mu, int * d_mf){
  
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int total_number_of_threads = blockDim.x*gridDim.x;
  int start = id;
#ifdef HYBRID
  if(id == 0)
    *d_mf = 0; //reinit edge size in frontier
#endif
  
  __syncthreads();
  for(int i = start; i < num_vertices; i += total_number_of_threads){ //iterate over vertices
    if(d_depth[i] == -1 && d_V[i] != -1){ //check if vertex does not have a parent
      int jend = d_V[i+1];
      bool notfound = true;
      for(int k = 2; jend == -1; k++){
	jend = d_V[i+k];
      }
      for(int j = d_V[i]; j < jend && notfound; j++){ //iterate over current vertices neighbors
	if(d_depth[d_A[j]] == (level - 1)){ //check if current vertex's neighbor is in current array
	  d_depth[i] = level;
	  *d_still_running = true;
	  notfound = false;
#ifdef HYBRID
	  //Collect heuristics
	  int edge_amount = jend - d_V[i];
	  atomicAdd(d_mf, edge_amount);
	  atomicAdd(d_mu, -1*edge_amount);
	  //
#endif
	}
      }
    }
  }
}
__global__ void TopDown(int num_vertices, int * d_V, int * d_A, int level, int * d_depth, bool * d_still_running, int * d_mu, int * d_mf){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int total_number_of_threads = blockDim.x*gridDim.x;
  int start = id;
#ifdef HYBRID
  if(id == 0)
    *d_mf = 0; //reinit edge size in frontier
#endif
  
 __syncthreads();
 for(int i = start; i < num_vertices; i += total_number_of_threads){
    if(d_depth[i] == (level - 1)){
      int jend = d_V[i+1];
      for(int k = 2; jend == -1; k++){ //eliminate isolated vertices
	jend = d_V[i+k];
      }
      for(int j = d_V[i]; j < jend; j++){ //iterate over current vertices neighbors
	if(d_depth[d_A[j]] == -1){ //check if the neigbor already has a parent
	  d_depth[d_A[j]] = level; //update depth
#ifdef HYBRID
	  //collect hueristics
	  int aend = d_V[d_A[j]+1];
	  for(int k = 2; aend == -1; k++){ //eliminate isolated vertices
	    aend = d_V[d_A[j]+k];
	  }
	  int edge_amount = aend - d_V[d_A[j]];
	  atomicAdd(d_mf, edge_amount);
	  atomicAdd(d_mu, -1*edge_amount);
	  //
#endif
    
	  *d_still_running = true;
        }
      }
    }
  }
}
int main(int argc, char *argv[]){
  if(argc!=2){
    printf("Make it work this way: ./<exec> <mtx file>");
    return 0;
  }
  cudaSetDevice(0);
  int num_threads = 1024;
  int num_blocks = 128;
  int * d_depth;
  int * d_mu;
  int * d_mf;
  int * d_V;
  int * d_A;
  bool * d_still_running; //Device still running bool
  int ** graph = read_mtx(argv[1]);
  int * V = graph[0];
  int * A = graph[1];
  int numVertices = graph[2][0];
  int numEdges = graph[2][1];
#ifdef DEBUGC
  printf("The num of vertices is %d \n ", numVertices);
  printf("The num of edges is %d \n ", numEdges);
#endif
  int source_vertex = 1; //(rand() % numVertices)+1;
  //Host pointers
  int * depth;
  int * mu;
  int * mf;
  bool * still_running; //Host still running bool
  int size = (numVertices+1) * sizeof(int);
  int e_size = numEdges * sizeof(int);
  cudaMalloc((void **)&d_mu, sizeof(int));
  cudaMalloc((void **)&d_mf, sizeof(int));
  cudaMalloc((void **)&d_depth, size);
  cudaMalloc((void **)&d_V, size);
  cudaMalloc((void **)&d_A, e_size);
  cudaMalloc((void **)&d_still_running, e_size);
  still_running = (bool *)malloc(sizeof(bool));
  mu = (int *)malloc(sizeof(int));
  mf = (int *)malloc(sizeof(int));
  depth = (int *)malloc(size);
  for(int i=0; i<numVertices; i++){
    depth[i] = -1;
  }
  depth[source_vertex] = 0;
  *mu = e_size;
  *mf = 0;
  
  cudaError_t rc = cudaMemcpy(d_depth, depth, size, cudaMemcpyHostToDevice);
  if (rc != cudaSuccess)
    printf("Could not allocate memory: %d", rc);
  cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A, sizeof(int) * numEdges, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mu, mu, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mf, mf, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_still_running, still_running, sizeof(bool), cudaMemcpyHostToDevice);
  
  int level = 1;
  //Time variables
  clock_t start, total;
  start = clock();
  *still_running = true;
  
  while(*still_running)
  {
    *still_running = false;
    cudaMemcpy(d_still_running, still_running, sizeof(bool), cudaMemcpyHostToDevice);
    //printf("LEVEL: %d, MU: %d, MF: %d, Alpha: %d, Result: %d.\n", level, *mu, *mf, ALPHA, *mf > (*mu / ALPHA));
#ifdef HYBRID
    if(*mf > (*mu / ALPHA)){
      BottomUp<<<num_blocks, num_threads>>>(numVertices, d_V, d_A, level, d_depth, d_still_running, d_mu, d_mf);
    }
    else{
      TopDown<<<num_blocks, num_threads>>>(numVertices, d_V, d_A, level, d_depth, d_still_running, d_mu, d_mf);
    }
#endif 
#ifdef BOTTOMUP
    BottomUp<<<num_blocks, num_threads>>>(numVertices, d_V, d_A, level, d_depth, d_still_running, d_mu, d_mf);
#endif 
#ifdef TOPDOWN
    TopDown<<<num_blocks, num_threads>>>(numVertices, d_V, d_A, level, d_depth, d_still_running, d_mu, d_mf);
#endif
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      printf("Level: %d, Error: %d\n", level, error);
    cudaMemcpy(still_running, d_still_running, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, d_mu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mf, d_mf, sizeof(int), cudaMemcpyDeviceToHost);
    level++;
  }
  total = clock() - start;
  cout << "The total running time is: "<< (float)(total)/CLOCKS_PER_SEC<< " secs" <<endl;
  #ifdef CHECK_DEPTH_RESULT
    cudaMemcpy(depth, d_depth, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "DEPTH RESULTS: " << endl;
    for(int i = 0; i < numVertices; i++){
      cout << i << ", " << depth[i] << endl;
    }
  #endif
}
