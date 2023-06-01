__global__ void sum(const double* xs, int size, double* results) {
extern __shared__ double shared_xs[];
int index = blockIdx.x * blockDim.x + threadIdx.x;
shared_xs[threadIdx.x] = (index < size) ? xs[index] : 0;
__syncthreads();
for (int stride =  blockDim.x/2; stride > 0; stride=stride/ 2) {
        if (stride > threadIdx.x) {
            shared_xs[threadIdx.x] = fmaxf(shared_xs[threadIdx.x + stride],shared_xs[threadIdx.x]);
        }

__syncthreads();
}
    
if (threadIdx.x == 0){
    if (blockIdx.x==0){
    results[blockIdx.x]= (shared_xs[0]);
    }
if (blockIdx.x>0){
    results[blockIdx.x]= fmaxf(shared_xs[0],results[blockIdx.x-1]);
}
}
}