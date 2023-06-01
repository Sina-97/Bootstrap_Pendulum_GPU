__global__ void normalise_ws(
   const double* ws,const double max_w, double* ws_norm)
{
 int index = blockDim.x * blockIdx.x + threadIdx.x;

    ws_norm[index] = exp(ws[index] - max_w);
}