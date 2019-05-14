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

using namespace std;



//===========Init bfs gpu=====================//
/*
void initBFSgpu(int * V, int * A, int startVertex, int *&depth, int *&parent, int numVertices, int numEdges){

int firstVertex = startVertex;


int size = numVertices * sizeof(int);
int e_size = numEdges * sizeof(int);

cudaMalloc((void **)&d_parent, size);
cudaMalloc((void **)&d_depth, size);
cudaMalloc((void **)&d_V, size);
cudaMalloc((void **)&d_A, e_size); 

parent = (int *)malloc(size); //random_ints(parent, numVertices);
depth = (int *)malloc(size); //random_ints(depth, numVertices);

parent[startVertex] = 0;
depth[startVertex] = 0;

cudaMemcpy(d_depth, depth, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_parent, parent, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_A, A, e_size, cudaMemcpyHostToDevice);


}
*/
//============================================//
//===============Finalize Cuda BFS============//
/*
void finalizeCudaBfs(int *&depth, int *&parent, int size) {

    //copy memory from device

    cudaMemcpy(parent, d_parent, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(depth, d_depth, size, cudaMemcpyDeviceToHost);

}
*/
//============================================//

//=================ParallelGPU================//
__global__
void TopDown(int num_vertices, int * d_V, int * d_A, int level, int *d_parent, int *d_depth, int &changed){

int tid = blockIdx.x * blockDim.x + threadIdx.x;
int CValue = 0;

//printf("The number of vertices: %d\n", num_vertices);
//printf("TID = %d \n", tid);

//printf("d_depth[0]%d\n", d_depth[0]);

if (tid < num_vertices /*&& d_depth[tid] == level*/){
	int u = tid; //changed=1;
	for(int i= d_V[u]; i< d_V[u+1]-d_V[u]; i++){
		int v = d_A[i];
		if(level + 1 < d_depth[v]){
	//	if(d_depth[v]==2147483647){
			d_depth[v] = level+1;
			d_parent[v] = i;			
			CValue = 1;
//printf("d_depth[v]= %d\n", d_depth[v]);
			}
//		}
	}

}
	if(CValue){
		//changed = CValue;
		d_parent[0] = CValue; 
	}
//	printf("CValue\n");

}

//============================================//

//===============ParallelCudaBFS==============//
/*
void simParallelCudaBFS(int * V, int * A, int source_vertex, int *&depth, int *&parent, int numVertices, int numEdges){

	printf("CHECK 2");
	
	//initBFSgpu(V, A, startVertex, depth, parent, numVertices, numEdges);
	
	//printf("source vertex %d", source_vertex);
	//int firstVertex = source_vertex;

int * d_depth;
int * d_parent;
int * d_V;
int * d_A;


printf("CHECK 3");

int size = numVertices * sizeof(int);
int e_size = numEdges * sizeof(int);


cudaMalloc((void **)&d_parent, size);
cudaMalloc((void **)&d_depth, size);
cudaMalloc((void **)&d_V, size);
cudaMalloc((void **)&d_A, e_size); 

//parent = (int *)malloc(size); //random_ints(parent, numVertices);
//depth = (int *)malloc(size); //random_ints(depth, numVertices);

parent[source_vertex] = 0;
depth[source_vertex] = 0;

printf("CHECK 3");

cudaMemcpy(d_depth, depth, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_parent, parent, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_A, A, e_size, cudaMemcpyHostToDevice);


	
	int *changed;
cudaMalloc((void **)&changed, sizeof(int));
	int level = 0;
	*changed = 1;

	while(*changed){
	
		*changed = 0;
		TopDown<<< 1, 6>>>(numVertices, d_V, d_A, level, d_parent, d_depth, changed);
		//cudaDeviceSynchronize();
		level++;
	}
	cudaDeviceSynchronize();
	
	cudaMemcpy(parent, d_parent, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(depth, d_depth, numVertices * sizeof(int), cudaMemcpyDeviceToHost);	
	
	printf("FINALIZED");
	//finalizeCudaBfs(distance, parent, size);

}
*/
//============================================//
//=================READ_MTX===================//

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
  int vertex_amount, max_vertex_id, tuple_amount,  edge_amount;
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


//=======================================//

//================Seq CPU BFS================//

void ** seq_bfs(int * v, int * a, int numVertices, int sourceNode){

        //int t = omp_get_wtime();
	  clock_t t = clock();
	//char * visited = new char[numVertices]();
        int *depth = new int[numVertices]();
        int * parent = new int[numVertices]();
        int *next=new int[numVertices];
        int * frontier=new int[numVertices];
        long fSize = 0; // size of the frontier queue
        long nSize = 1; // size of the next queue
        int * depthCount = new int[numVertices]();
        frontier[fSize++]=sourceNode;
        depth[sourceNode] = 1;
        depthCount[1]=1;
        parent[sourceNode] = -1;

#define DEBUG
        int step=2;
        for (; nSize!=0;step++){

#ifdef DEBUG
                cout << " step " << step<< endl;
#endif
                nSize=0;
                for (int i =0; i<fSize; i++){
                        int ver = frontier[i];
			if (v[ver]==-1)continue;

#ifdef DEBUG
                   //     cout << "fr " << ver;
#endif
                        int numOfNeigh=v[ver+1]-v[ver];
			int j=0;
			while(numOfNeigh<0){
				if (ver+2+j>=numVertices-1){
					cout << "wtf v="<< ver << " v[ver+1+j]="<<v[ver+1+j]<<" j="<<j<<endl;
				}
				numOfNeigh=v[ver+1+(++j)]-v[ver];

			} 
			for (int j =0; j<numOfNeigh; j++){
				int n = a[v[ver]+j]; // the neighbor index
				if (n>numVertices-1){
					cout << "wtf broooo ver "<< ver << " j " <<j <<" v[ver] " << v[ver] << "v[ver]+1 " << v[ver]+1 << " a[ver[ver]+1] " << a[v[ver]+j] << endl;			
				}					

#ifdef DEBUG                                
		///		cout << " n " << n ;
#endif
                                if (depth[n]=='\0'){
#ifdef DEBUG
                                       // cout << " added ";
#endif
                                        depth[n]=step;
					parent[n] = ver;
					if(nSize>=numVertices-1) printf("holdup");
                                        next[nSize++]=n;
                                        depthCount[step]++;
                                }
#ifdef DEBUG
				//else
					 //cout << " not added ";

                ///        cout << endl;

#endif
                }
		}

#ifdef DEBUG

                cout <<"depth "<< depthCount[step];
                cout << " nSize " << nSize<< endl;

#endif
                int * temp = next;
                next = frontier;
                frontier = temp;
                fSize = nSize;
                //for (int k =0; k<nSize;k++) frontier.push_back(next[k]);
        }

        depthCount[0] = step-1;
	cout << "Time to bfs " << ((float)clock()-t)/CLOCKS_PER_SEC<< endl;
//      for (int l = 0; l<numVertices;l++) cout <<(int)depth[l] << endl;
//      for (int l = 0; l<numVertices;l++) cout <<(int)parent[l] << endl;
        delete []frontier;
        delete []next;
        void ** ret = new void *[3];
        ret[0] = depth;
        ret[1] = parent;
        ret[2] = depthCount;
        return ret;
}



//=======================================//


int main(int argc, char *argv[]){

	if(argc!=2){
		printf("Make it work this way: ./<exec> <mtx file>");
		return 0;
	}

	int * d_depth ;
	int * d_parent;
	int * d_V;
	int * d_A;


	int ** graph = read_mtx(argv[1]);
	int * V = graph[0];
	int * A = graph[1];
        int numVertices = graph[2][0];
	int numEdges = graph[2][1];	
	printf("The num of vertices is %d \n ", numVertices);
	printf("The num of edges is %d \n ", numEdges);
	
	void ** AllINeed = new void *[3];
	int source_vertex = (rand() % numVertices)+1;
	
	//Host pointers
	int * depth;
	int * parent;
	int size = (numVertices+1) * sizeof(int);
	printf("EKHDEM YA KHRA \n");
	int e_size = numEdges * sizeof(int);


	cudaMalloc((void **)&d_parent, size);
	cudaMalloc((void **)&d_depth, size);
	cudaMalloc((void **)&d_V,  size);
	cudaMalloc((void **)&d_A, e_size); 

		
	parent = (int *)malloc(size); //random_ints(parent, numVertices);
	depth = (int *)malloc(size); //random_ints(depth, numVertices);
	for(int i=0; i<numVertices; i++){
		depth[i] = std::numeric_limits<int>::max();
		parent[i] = std::numeric_limits<int>::max();
		//depth[i] = 0; parent[i] = 0;
	}
	parent[source_vertex] = 0;
	depth[source_vertex] = 0;


	cudaError_t rc = cudaMemcpy(d_depth, depth, size, cudaMemcpyHostToDevice);
	if (rc != cudaSuccess)
		printf("Could not allocate memory: %d", rc);
	cudaMemcpy(d_parent, parent, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, A, sizeof(int) * numEdges, cudaMemcpyHostToDevice);

	
	//cudaDeviceSynchronize();

	//printf("HERE YA ZA7I\n");
	//printf("ZA777III1\n");
	printf("ZA777III2\n");
	//printf("d_depth[111] = %d\n", d_parent[111]);
	int changed;
	cudaMalloc((void **)&changed, sizeof(int));
	int level = 0;
	changed = 1;
	//int changed_host;
	//changed_host = (int)malloc(sizeof(int));
	//changed_host = 1;
	//cudaMemcpy(*changed, changed_host, sizeof(int), cudaMemcpyHostToDevice);
	parent[0] = 1;
int iter =0;
//	while(changed){
	while(parent[0]==1){
//		printf("Entered here\n");
		//changed = 0;
		parent[0] = 0;
		//changed_host = 0;
		TopDown<<<1, 10>>>(numVertices, d_V, d_A, level, d_parent, d_depth, changed);
		
		cudaMemcpy(parent, d_parent, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(depth, d_depth, numVertices * sizeof(int), cudaMemcpyDeviceToHost);	

		cudaDeviceSynchronize();
		//cudaMemcpy(changed_host, *changed, sizeof(int), cudaMemcpyDeviceToHost);
		level++;
//changed++;
iter++; if(iter==50) break;		
//printf("Level = %d\n", level);
	}
	
	
	
	//printf("FINITTA\n");

	//cudaMemcpy(parent, d_parent, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
       // cudaMemcpy(depth, d_depth, numVertices * sizeof(int), cudaMemcpyDeviceToHost);	
	//for(int i =0; i<numVertices; i++)
	printf("parent[i] = : %d\n", depth[1000]);

	printf("FINALIZED\n");
	//finalizeCudaBfs(distance, parent, size);

	

	//========================================================//
	/*	
	//initBFSgpu(V, A, source_vertex, depth, parent, numVertices, numEdges);
	printf("PRE-CHECK\n");
	printf("source: %d \t numVertices: %d \t numEdges: %d \n", source_vertex, numVertices, numEdges);
	simParallelCudaBFS(V, A, source_vertex, depth, parent, numVertices, numEdges);
	printf("depth[source_vertex] %d \n", depth[source_vertex]);
	//AllINeed = seq_bfs(V, A, numVertices, source_vertex);
	//cudaDeviceSynchronize();
	*/
	



}
