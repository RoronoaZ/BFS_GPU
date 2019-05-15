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
#include <set>

using namespace std;

#define MINCAPACITY	65535

///////////////////////////////////////////////

template <class T>
__device__ __host__ inline T my_fetch_add(T *ptr, T val) {
#ifdef OPENMP
#ifdef EXT
	return __sync_fetch_and_add(ptr,val);
#endif
#ifdef OPENMP2
	T old;
	#pragma omp atomic capture
	{old = *ptr; *ptr += val;}
	return old;
#endif
#else
	T old; old = *ptr; *ptr += val;
	return old;
#endif
}

template <class T>
__device__ __host__ inline T my_fetch_sub(T *ptr, T val) {
#ifdef OPENMP
#ifdef GCC_EXTENSION
	return __sync_fetch_and_sub(ptr,val);
#endif
#ifdef OPENMP2
	T old;
	#pragma omp atomic capture
	{old = *ptr; *ptr -= val;}
	return old;
#endif
#else
	T old; old = *ptr; *ptr -= val;
	return old;
#endif
};


typedef struct vert_queue {

	__device__ __host__ int intervalPush(int *start, int nitems);
	__device__ __host__ int push(int work);
	int intervalPop(int *start, int nitems);
	int pop(int &work);
	void clear();
	void myItems(int &start, int &end);
	__device__ __host__ int getItem(int at);

	void initialization();
	void init(int initialcapacity);
	void setSize(int hsize);
	int getSize();
	void setCapacity(int hcapacity);
	int getCapacity();
	void setInitialSize(int hsize);
	int calculateSize(int hstart, int hend);
	void copyToNew(int *olditems, int *newitems, int oldsize, int oldcapacity);

	int reserveMem(int space);
	int *alloc(int allocsize);
	int realloc(int space);
	int * data();

	int freeSize();
	int *items;
	int start, end;
	int capacity;
	int noverflows;
	vert_queue();
	~vert_queue();

} vert_queue;

vert_queue::vert_queue() {
	initialization();
}

void vert_queue::initialization() {
	init(0);
}



void vert_queue::init(int initialcapacity) {
	setCapacity(initialcapacity);
	setInitialSize(0);
	items = NULL;
	if (initialcapacity) items = alloc(initialcapacity);
	noverflows = 0;
}

int *vert_queue::alloc(int allocsize) {
	int *ptr = NULL;
	if(allocsize > 0)
		ptr = (int *)malloc(allocsize * sizeof(int));
	if(ptr == NULL)
		printf("%s(%d): Allocating %d failed.\n", __FILE__, __LINE__, allocsize);
	return ptr;
}

int vert_queue::getCapacity() {
	return capacity;
}

int vert_queue::calculateSize(int hstart, int hend) {
	if (hend >= hstart) {
		return hend - hstart;
	}
	// circular queue.
	int cap = getCapacity();
	return hend + (cap - hstart + 1);
}

int vert_queue::getSize() {
	return calculateSize(start, end);
}

void vert_queue::setCapacity(int cap) {
	capacity = cap;
}

void vert_queue::setInitialSize(int size) {
	start = 0;
	end = 0;
}

void vert_queue::setSize(int size) {
	int cap = getCapacity();
	if (size > cap) {
		printf("%s(%d): buffer overflow, setting size=%d, when capacity=%d.\n", __FILE__, __LINE__, size, cap);
		return;
	}
	if (start + size < cap) {
		end   = start + size;
	} else {
		size -= cap - start;
		end   = size;
	}
}

void vert_queue::copyToNew(int *olditems, int *newitems, int oldsize, int oldcapacity) {
	if (start < end) {	// no wrap-around.
		memcpy(newitems, olditems + start, oldsize * sizeof(int));
	} else {
		memcpy(newitems, olditems + start, (oldcapacity - start) * sizeof(int));
		memcpy(newitems + (oldcapacity - start), olditems, end * sizeof(int));
	}
}

int vert_queue::realloc(int space) {
	int cap = getCapacity();
	int newcapacity = (space > MINCAPACITY ? space : MINCAPACITY);
	if (cap == 0) {
		setCapacity(newcapacity);
		items = alloc(newcapacity);
		if (items == NULL) {
			return 1;
		}
		//printf("\tworklist capacity set to %d.\n", getCapacity());
	} else {
		int *itemsrealloc = alloc(newcapacity);
		if (itemsrealloc == NULL) {
			return 1;
		}
		int oldsize = getSize();
		copyToNew(items, itemsrealloc, oldsize, cap);
		//dealloc();
		free(items);
		setInitialSize(0);

		items = itemsrealloc;
		setCapacity(newcapacity);
		start = 0;
		end = oldsize;
		printf("\tworklist capacity reset to %d.\n", getCapacity());
	}
	return 0;
}

int vert_queue::freeSize() {
	return getCapacity() - getSize();
}

int vert_queue::reserveMem(int space) {
	if (freeSize() >= space) {
		return 0;
	}
	realloc(space);
	return 1;
}


vert_queue::~vert_queue() {
}

__device__ __host__ int vert_queue::intervalPush(int *copyfrom, int nitems) {
	if (copyfrom == NULL || nitems == 0) return 0;

	int lcap = capacity;
	int offset = my_fetch_add<int>(&end, nitems);
	if (offset >= lcap) {	// overflow.
		my_fetch_sub<int>(&end, nitems);
		return 1;
	}
	for (int ii = 0; ii < nitems; ++ii) {
		items[(offset + ii) % lcap] = copyfrom[ii];
	}
	return 0;
}

__device__ __host__ int vert_queue::push(int work) {
	return intervalPush(&work, 1);
}

int vert_queue::intervalPop(int *copyto, int nitems) {
	int currsize ;
	if (end >= start) {
		currsize = end - start;
	} else {
		currsize =  end + (capacity - start + 1);
	}

	if (currsize < nitems) {
		nitems = currsize;
	}
	int offset = 0;
	int lcap = capacity;
	if (nitems) {
		if (start + nitems < lcap) {
			offset = my_fetch_add<int>(&start, nitems);
		} else {
			offset = my_fetch_add<int>(&start, start + nitems - lcap);
		}
	}
	// copy nitems starting from offset's index.
	for (int ii = 0; ii < nitems; ++ii) {
		copyto[ii] = items[(offset + ii) % lcap];
	}
	return nitems;
}

int vert_queue::pop(int &work) {
	return intervalPop(&work, 1);
}

void vert_queue::clear() {
	setSize(0);
}


__device__ __host__ int vert_queue::getItem(int at) {
	int size;
	if (end >= start) {
		size = end - start;
	} else {
		size =  end + (capacity - start + 1);
	}


	if (at < size) {
		return items[at];
	}
	return -1;

}


//===========Init bfs gpu=====================//
//============================================//


//===============Finalize Cuda BFS============//
//============================================//

//=================Kernel================//
__global__
void TopDown(int num_vertices, int * d_V, int * d_A, int level, int *d_parent, int *d_depth, vert_queue * d_input_q_ptr, vert_queue * d_R_q){

int tid = blockIdx.x * blockDim.x + threadIdx.x;
int CValue = 0;

//printf("The number of vertices: %d\n", num_vertices);
//printf("TID = %d \n", tid);

//printf("d_depth[0]%d\n", d_depth[0]);

if (tid < num_vertices /*&& d_depth[tid] == level*/){
	unsigned u = tid; //changed=1;
//printf("vertex = %d \n", u);		
	
	int vertex  = d_input_q_ptr->items[u];
printf("vertex = %d\n", vertex);

	int vertexx = d_input_q_ptr->items[u+1];
	//for(int i= vertex; i< (vertexx-vertex); i++){
	for(int i=d_V[vertex]; i<d_V[vertexx]-d_V[vertex]; i++){
		int v = d_A[i];

		if(level + 1 < d_depth[v]){
			d_depth[v] = level+1;
			d_parent[v] = i;			
			CValue = 1;
			d_R_q->push(v);
			}
//		}
	
	}

}
	if(CValue){
		//changed = CValue;
		d_parent[0] = CValue; 
	}
//	printf("OUT\n");

}

//============================================//

//===============ParallelCudaBFS==============//
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

//================Vert_queue================//


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
	int size = (numVertices) * sizeof(int);
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

//========================================================//

vert_queue input_q, output_q, *input_q_ptr, *R_q, *tmp;
vert_queue d_input_q, d_output_q, *d_input_q_ptr, *d_R_q;


	input_q.reserveMem(numVertices);
	output_q.reserveMem(numEdges);

	/*
	d_input_q_ptr = new vert_queue();
	d_R_q = new vert_queue();

	d_input_q_ptr->items = (int*)malloc(sizeof(int)*numVertices);
	d_R_q->items = (int*)malloc(sizeof(int)*numVertices);
	
	*/

	input_q_ptr = &input_q;
	R_q = &output_q;
		



	int *range = (int *)malloc(numVertices * sizeof(unsigned));
	for (unsigned j = 0; j < numVertices; j++)
		range[j] = j;
	//for(int i =0; i<numVertices; i++)
	//	printf("V[%d]=%d \n", V[i]);
	input_q.intervalPush(range, numVertices);
	
	int frontier = input_q.getSize();
	//printf("%d\n",input_q_ptr->getItem(556));	
	//printf("item[i]=%d\n", input_q_ptr->items[3]);


	

//========================================================//
	

	//cudaDeviceSynchronize();

	printf("ZA777III2\n");
	int changed;
	cudaMalloc((void **)&changed, sizeof(int));
	int level = 0;
	changed = 1;
	int size_struct = input_q.getSize() * sizeof(int);

/************
	cudaError_t rt = cudaMalloc(&d_input_q_ptr->items, size);
	if(rt != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(rt));
	cudaMalloc(&d_R_q->items, e_size);
	printf("HHHHHHHHHHHHH\n");
	//cudaMalloc(&d_R_q->items, e_size);
************/



// input_q		
int * h_arr; 
int * d_arr;
h_arr = new int[numVertices]();
cudaMalloc((void **)&d_arr, numVertices * sizeof(*d_arr));
cudaMalloc((void **)&d_input_q_ptr, sizeof(*d_input_q_ptr));

cudaMemcpy(d_arr, h_arr, size * sizeof(*d_arr), cudaMemcpyHostToDevice);
cudaMemcpy(&(d_input_q_ptr->items), &d_arr, sizeof(d_input_q_ptr->items), cudaMemcpyHostToDevice);


//=================//	
//R_q
int * h_arr1;
int * d_arr1;
h_arr1 = new int[numEdges]();
cudaMalloc((void **)&d_arr1, numEdges * sizeof(*d_arr1));
cudaMalloc((void **)&d_R_q, sizeof(*d_R_q));

cudaMemcpy(d_arr1, h_arr1, e_size * sizeof(*d_arr1), cudaMemcpyHostToDevice);
cudaMemcpy(&(d_R_q->items), &d_arr1, sizeof(d_R_q->items), cudaMemcpyHostToDevice);
//=================//


		cudaMemcpy(d_depth, depth, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_parent, parent, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_A, A, sizeof(int) * numEdges, cudaMemcpyHostToDevice);


//	while(changed){
//	while(parent[0]==1){
	while(frontier){
//		printf("Entered here\n");
		//changed = 0;
		//parent[0] = 0;
		//changed_host = 0;
	
//printf("CHECK 2\n");

		//////////

	

		//cudaMalloc((void **)&d_input_q_ptr, sizeof(vert_queue));
		//cudaMalloc((void **)&d_R_q, sizeof(vert_queue));
	
//printf("CHECK 21\n");
	/**********
		cudaMemcpy(d_input_q_ptr, input_q_ptr, sizeof(vert_queue*), cudaMemcpyHostToDevice);
		cudaError_t rc = cudaMemcpy(d_input_q_ptr->items, input_q_ptr->items, sizeof(int)*numVertices, cudaMemcpyHostToDevice);
		d_input_q_ptr = input_q_ptr;
		d_input_q_ptr->items = input_q_ptr->items;
	
//printf("CHECK 22\n");
		if(rc != cudaSuccess)
			printf("Error in mem, The Error: %s\n", cudaGetErrorString(rc));
		cudaError_t rc1 = cudaMemcpy(d_R_q, R_q, sizeof(vert_queue*), cudaMemcpyHostToDevice);
		if(rc1 != cudaSuccess)
			printf("Error in mem2\n");
		cudaMemcpy(d_R_q->items, R_q->items, e_size, cudaMemcpyHostToDevice);
	***********/
		//////////
//printf("CHECK 3\n");
	
		TopDown<<<10, 256>>>(numVertices, d_V, d_A, level, d_parent, d_depth, d_input_q_ptr, d_R_q);
		cudaDeviceSynchronize();
		

		cudaMemcpy(h_arr, d_arr, size * sizeof(*h_arr), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_arr1, d_arr1, e_size * sizeof(*h_arr1), cudaMemcpyDeviceToHost);

		cudaMemcpy(R_q, d_R_q, sizeof(*d_R_q), cudaMemcpyDeviceToHost);
		
		//printf("R_q %d \n", R_q->getItem(3));
		//printf("h_arr1[450]=%d\n", h_arr1[450]);
//printf("CHECK 4\n");
	/**********	
		//////////
		cudaMemcpy(&input_q_ptr, d_input_q_ptr, sizeof(int)*numVertices, cudaMemcpyDeviceToHost);
		cudaMemcpy(&R_q, d_R_q, sizeof(int)*numEdges, cudaMemcpyDeviceToHost);
	**********/
//printf("CHECK 5\n");
		
		frontier = R_q->getSize();
		tmp = input_q_ptr; input_q_ptr = R_q; R_q = tmp;
		R_q->clear();
		//////////
		
		//cudaMemcpy(changed_host, *changed, sizeof(int), cudaMemcpyDeviceToHost);
		level++;
//changed++;
printf("Level = %d\n", level);
	}
	
		cudaMemcpy(parent, d_parent, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(depth, d_depth, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
	
	
	//printf("FINITTA\n");

	//cudaMemcpy(parent, d_parent, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
       // cudaMemcpy(depth, d_depth, numVertices * sizeof(int), cudaMemcpyDeviceToHost);	
	//for(int i =0; i<numVertices; i++)
	//	if(depth[i]!=2147483647)	
	//		printf("depth[%d] = : %d\n", i,depth[i]);

	printf("FINALIZED\n");
	//finalizeCudaBfs(distance, parent, size);

	cudaFree(d_arr1); cudaFree(d_arr); cudaFree(d_input_q_ptr);	

	//========================================================//
	



}
