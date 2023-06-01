__global__ void loglike(
    double ydata,const double* x1_n,double mean1, double stdev,
    double* ll)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // log(Z * exp(arg)) = log(Z) + log(exp(arg)) = log(Z) + arg
    double var = stdev * stdev; // sigma^2
    double Z = 1 / sqrt(2*3.14 * var);
    double diff = (ydata-sin(x1_n[index])) - mean1;
    double arg = -0.5 * diff * diff / var; 

    ll[index] =(log(Z) + arg);
}