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

#define DEBUGC

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
  cout << "Tuple Matrix is created. Ex. M[0][0] = " << M[0][0] << "M[0][1] = " << M[0][1] << endl;
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
__global__ void BottomUp(int num_vertices, int * d_V, int * d_A, int 
level, int *d_parent, int *d_depth, int * d_current, int * frontier){
  
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int searchAmount = (num_vertices / (blockDim.x*gridDim.x)) + 1;
  int start = id*searchAmount;
  int end = (id+1)*searchAmount;
  if(end > num_vertices){
    end = num_vertices;
  }
  if(id == 0){
    for(int i=0; i < num_vertices; i++){ //reinit frontier
      frontier[i] = -1;
    }
  }
  //printf("Thread %d, start %d, end %d\n", id, start, end);
  
  __syncthreads();
  for(int i = start; i < end; i++){ //iterate over vertices
    if(d_parent[i] == -1){ //check if vertex does not have a parent
      bool notfound = true;
      for(int j = d_V[i]; j < d_V[i+1] && notfound; j++){ //iterate over current vertices neighbors
	if(d_current[d_A[j]] == 1){ //check if current vertex's neighbor is in current array
	  //some kind of locking mechanism is needed for thread size bigger than 32 i guess
	  d_parent[i] = d_A[j]; //update parent information
	  d_depth[i] = level;
	  frontier[i] = 1; //add vertex to frontier
	  //printf("Thread %d, vertex %d, neigh %d\n", id, i, d_A[j]);
	  notfound = false;
	}
      }
    }
  }
  
  __syncthreads();
  if(id == 0){
    for(int i = 0; i < num_vertices; i++){ // another swap idea
      d_current[i] = frontier[i];
    }
    //d_current = frontier; //swap current with frontier
  }
}
__global__ void TopDown(int num_vertices, int * d_num_current, int * 
d_V, int * d_A, int level, int *d_parent, int *d_depth, int * d_current, 
int * frontier, bool * d_still_running){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  //if(id == 0)
  //printf("\nLEVEL %d, NUM CURR %d.\n", level, d_num_current[0]);
  int searchAmount = (num_vertices / (blockDim.x*gridDim.x)) + 1;
  int start = id*searchAmount;
  int end = (id+1)*searchAmount;
  if(end > num_vertices){
    end = num_vertices;
  }
  /*if(id == 0){ // not needed for level check algorithm
     for(int i=0; i < num_vertices; i++){ //reinit frontier
      frontier[i] = -1;
    }
    }*/
 
  *d_still_running = false;

  __syncthreads();
  //d_num_current[0] = 0;
  //printf("Start %d, End %d\n.", start, end);
  for(int i = start; i < end; i++){ //iterate over current array
    if(d_current[i] == (level - 1)){
      //printf("i, %d, jstart: %d, jend: %d.\n", i, d_V[i], d_V[i+1]);
      for(int j = d_V[i]; j < d_V[i+1]; j++){ //iterate over current vertices neighbors
        if(d_parent[d_A[j]] == -1){ //check if the neigbor already has a parent

	  *d_still_running = true;
	  //printf("Vertex %d, neihg %d\n.", d_current[i], d_A[j]);
	  d_parent[d_A[j]] = i; //update parent
	  d_depth[d_A[j]] = level; //update depth
	  frontier[d_A[j]] = level;
	  //frontier[d_num_current[0]] = d_A[j]; //update frontier
	  //d_num_current[0]++;
        }
      }
    }
  }
  
   __syncthreads();
   if(id == 0){
    for(int i = 0; i < num_vertices; i++){ // another swap idea
      d_current[i] = frontier[i];
      }
    //d_current = frontier; //swap current with frontier
    /*printf("Level: %d.\n", level);
    printf("DEVICE: \n");
    int lil = 0;
    for(int i = 0; i < num_vertices; i++){
      if(d_current[i] != -1)
	lil++;
	//printf("%d, %d\n", i, d_current[i]);
    }
    printf("Amount: %d\n.", lil);
    printf("\n");*/
  }
}
int main(int argc, char *argv[]){
  if(argc!=2){
    printf("Make it work this way: ./<exec> <mtx file>");
    return 0;
  }
  cudaSetDevice(0);
  int num_threads = 256;
  int * d_depth ;
  int * d_parent;
  int * d_V;
  int * d_A;
  //bool * d_still_running;
  int * d_current;
  int * d_num_current;
  int * d_frontier;
  int ** graph = read_mtx(argv[1]);
  int * V = graph[0];
  int * A = graph[1];
  int numVertices = graph[2][0];
  int numEdges = graph[2][1];

  int num_blocks = 1;

#ifdef DEBUGC
  printf("The num of vertices is %d \n ", numVertices);
  printf("The num of edges is %d \n ", numEdges);
#endif
  int source_vertex = 1; //(rand() % numVertices)+1;
  //Host pointers
  int * depth;
  int * parent;
  int * current;
  int * num_current;
  int * frontier;
  
  //Host still running bool
  bool * still_running;
  //Device still running bool
  bool * d_still_running;

  int size = (numVertices+1) * sizeof(int);
  int e_size = numEdges * sizeof(int);
  cudaMalloc((void **)&d_num_current, sizeof(int));
  cudaMalloc((void **)&d_current, size);
  cudaMalloc((void **)&d_frontier, size);
  cudaMalloc((void **)&d_parent, size);
  cudaMalloc((void **)&d_depth, size);
  cudaMalloc((void **)&d_V, size);
  cudaMalloc((void **)&d_A, e_size);

  //d_still_running
  cudaMalloc((void **)&d_still_running, e_size);

#ifdef DEBUGC
  printf("Allocated device memory.");
#endif
  
  parent = (int *)malloc(size);
  depth = (int *)malloc(size);
  current = (int *)malloc(size);
  num_current = (int *)malloc(sizeof(int));
  frontier = (int *)malloc(size);
  for(int i=0; i<numVertices; i++){
    depth[i] = -1;
    parent[i] = -1;
    current[i] = -1;
    frontier[i] = -1;
  }
  num_current[0] = 1;
  //current[source_vertex] = 1; for bottom up
  current[source_vertex] = 0;
  parent[source_vertex] = 0;
  depth[source_vertex] = 0;
  
  still_running = (bool *)malloc(e_size);

  #ifdef DEBUGC
  cout << "Current, parent and depth initializef" << endl;
  #endif
  
  cudaError_t rc = cudaMemcpy(d_depth, depth, size, 
cudaMemcpyHostToDevice);
  if (rc != cudaSuccess)
    printf("Could not allocate memory: %d", rc);
  cudaMemcpy(d_parent, parent, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A, sizeof(int) * numEdges, cudaMemcpyHostToDevice);
  cudaMemcpy(d_current, current, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_num_current, num_current, sizeof(int), 
cudaMemcpyHostToDevice);
  cudaMemcpy(d_frontier, frontier, size, cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_still_running, still_running, e_size, cudaMemcpyHostToDevice);
 
  #ifdef DEBUGC
  cout << "Mem copied from host to device." << endl;
  #endif
  
  int level = 1;
  /*while (level < 6)
  {3
    BottomUp<<<num_blocks, num_threads>>>(numVertices, d_V, d_A, level, 
d_parent, d_depth, d_current, d_frontier, d_still_running);
    cudaMemcpy(current, d_current, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      printf("Error: %d\n", error);
   
    #ifdef DEBUGC
    cout << "LEVEL: " << level << endl;
    cout << "HOST:" << endl;
    for(int i = 0; level == 5 && i < numVertices; i++){
      if(current[i] == 1)
	cout << i << ", " << current[i] << endl;
    }
    cout << endl;;
    #endif
    level++;
  }*/

  //Time variables
  clock_t start, total;
  start = clock();
//  while (level < 20)
  *still_running = true;
  while (*still_running)
  {
    TopDown<<<num_blocks, num_threads>>>(numVertices, d_num_current, 
d_V, d_A, level, d_parent, d_depth, d_current, d_frontier, d_still_running);
    cudaMemcpy(current, d_current, size, cudaMemcpyDeviceToHost);
    //cudaThreadSynchronize();
    
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      printf("Error: %d\n", error);
   
    #ifndef DEBUGC
    cout << "LEVEL: " << level << endl;
    cout << "HOST:" << endl;
    for(int i = 0; i < numVertices; i++){
      if(current[i] != -1){
	cout << i << ", " << current[i] << endl;
      }
      else{
	break;
      }
    }
    cout << endl;;
    #endif

    cudaMemcpy(still_running, d_still_running, sizeof(int)* numEdges, cudaMemcpyDeviceToHost);
    level++;
  }

  total = clock() - start;
  cudaMemcpy(parent, d_parent, numVertices * sizeof(int), 
cudaMemcpyDeviceToHost);
  cudaMemcpy(depth, d_depth, numVertices * sizeof(int), 
cudaMemcpyDeviceToHost);

cout << "The total running time is: "<< (float)(total)/CLOCKS_PER_SEC<< " secs" <<endl;

  cout << "DEPTH RESULTS: " << endl;
  for(int i = 0; i < numVertices; i++){
    cout << i << ", " << depth[i] << endl;
  }

}
