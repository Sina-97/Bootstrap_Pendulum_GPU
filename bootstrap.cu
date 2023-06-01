__global__ void loglike(
    const double* x1_prev,const double* x2_prev,const double* q_n1,const double* q_n2,double dt,double* x1_next,int J)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    double g=9.8;
    x1_next[index]=x1_prev[index]+x2_prev[index]*dt+q_n1[index];
    x1_next[index+J]=x2_prev[index]-g*sin(x1_prev[index])*dt+q_n2[index];
    
}