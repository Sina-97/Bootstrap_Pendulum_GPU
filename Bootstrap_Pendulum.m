clc
close all
clear all
dt = 0.01;
q_c = 0.1;
g = 9.81;

rSigma = 0.05;
Q = [ q_c * dt^3 / 3, q_c * dt^2 / 2;...
      q_c * dt^2 / 2, q_c * dt];
  
L = chol( Q );

T = 400; % Time horizon

x_data = zeros( 2, T);
x_data( 1, 1) = 0;
x_data( 2, 1) = 1;

for n = 2:T
    
    % Generate the next states with Eq.(5)
    xPrev = x_data( :, n - 1);
    
    x1_prev = xPrev(1);
    x2_prev = xPrev(2);
    
    w_n = randn( 2, 1);
    q_n = L * w_n;
    
    x1_next = x1_prev + x2_prev * dt + q_n(1);
    x2_next = x2_prev - g * sin( x1_prev ) * dt + q_n(2);
    
    xNext = [ x1_next; x2_next];
    x_data( :, n) = xNext;
end

x1 = x_data( 1, :);
x2 = x_data( 2, :);

y_data = sin( x1 ) + rSigma * randn( 1, T);

ydata = y_data;
xdata = x_data;

rSigma = 0.05;
rSigma2 = rSigma * rSigma;
N = length( ydata ); %
j=0;
% for J=10:100:10000
J=10000; %Number of Particles
% j=j+1;

% Eq.(6) lecture 6
Q = [ q_c * dt^3 / 3, q_c * dt^2 / 2;...
    q_c * dt^2 / 2, q_c * dt];
L=chol(Q);

x = zeros( 2, N);
x( 1, 1) = 0;
x( 2, 1) = 1;
mean1=0;

ws=zeros([1,N],'double');
ws_input=gpuArray(ws);

blockSize = 16;
numblock=ceil(J/blockSize);

Max_kernel = parallel.gpu.CUDAKernel('Max.ptx', 'Max.cu');
Max_kernel.ThreadBlockSize = [blockSize, 1, 1];
Max_kernel.GridSize = [numblock, 1, 1];
Max_kernel.SharedMemorySize = blockSize * 4; % 4 bytes per float

normalize_kernel = parallel.gpu.CUDAKernel('normalize.ptx', 'normalize.cu');
normalize_kernel.ThreadBlockSize = [blockSize, 1, 1];
normalize_kernel.GridSize = [numblock, 1, 1];

bootstrap_kernel = parallel.gpu.CUDAKernel('bootstrap.ptx', 'bootstrap.cu');
bootstrap_kernel.ThreadBlockSize = [blockSize, 1, 1];
bootstrap_kernel.GridSize = [numblock, 1, 1];

loglike_kernel = parallel.gpu.CUDAKernel('loglike.ptx', 'loglike.cu');
loglike_kernel.ThreadBlockSize = [blockSize, 1, 1];
loglike_kernel.GridSize = [numblock, 1, 1];

%% 1 Computing log-likelihood of the real data with cpu

output_y(1)=0;
for n = 2:N
    
    % Generate the next states with Eq.(5)
    xPrev = x( :, n - 1);
    
    x1_prev = xPrev(1);
    x2_prev = xPrev(2);
    
    w_n = randn( 2, 1);
    q_n = L * w_n;
    
    x1_next = x1_prev + x2_prev * dt + q_n(1);
    x2_next = x2_prev - g * sin( x1_prev ) * dt + q_n(2);
    
    
    xNext = [ x1_next; x2_next];
    x( :, n) = xNext;
end

x1 = x( 1, :);
x2 = x( 2, :);

y = sin( x1 );
eps=gpuArray(ydata-y);

%% 2 GPU Computation

x1_mu = 0.5;
x1_sigma = 1;

x2_mu = 0;
x2_sigma = 1;

X1 = zeros( J, N);
X2 = zeros( J, N);

% Step 1 of the bootstrap filter
X1(:,1) = x1_mu + x1_sigma * randn( J, 1);
X2(:,1) = x1_mu + x2_sigma * randn( J, 1);
tic
for n = 2:N
    
    x1_prev = X1( :, n - 1);
    x2_prev = X2( :, n - 1);
    y_n = y(n);
    
    ll_n = zeros( J, 1);
    w_n = randn( 2, J);
    q_n= L * w_n;
    q_n1=q_n(1,:);q_n2=q_n(2,:);
    x1_next=gpuArray(zeros(J,2));
    x2_next=gpuArray(zeros(J,1));
    
    % Step 2 - bootstrap filter propagation
    result=feval(bootstrap_kernel,gpuArray(x1_prev),gpuArray(x2_prev),gpuArray(q_n1),gpuArray(q_n2),dt,x1_next,J);
    x1_n=gather(result(:,1));
    x2_n=gather(result(:,2));
    
    X1( :, n) = x1_n;
    X2( :, n) = x2_n;
    
    eps=zeros(J,1);
    ll=zeros(J,1);
    
    % Step 3 - computing log-likelihood
    ll_result=feval(loglike_kernel,y_n,gpuArray(x1_n),mean1,rSigma,gpuArray(ll));
    ll_n=gather(ll_result);
    
    % Step 4 - weight normalization
    ws_norm=gpuArray(-Inf([numblock 1],'double'));
    maxW_n = feval(Max_kernel, gpuArray(ll_n),numel(ll_n), ws_norm );
    max_ws=(gather(maxW_n));
    max_ws=max(max_ws);
    
    ws_normalize=gpuArray(zeros([1 J],'double'));
    normal_ws = feval(normalize_kernel, gpuArray(ll_n),max_ws,ws_normalize );
    ws_ll=gather(normal_ws);
    weights = ws_ll / sum( ws_ll );
    
    %   Step 5 - resampling the states x_n
    resampleInds = resampleSystematic( weights )';
    X1( :, n) = X1( resampleInds, n);
    X2( :, n) = X2( resampleInds, n);
end
els_t=toc

% After this point we are just plotting the results
x1_groundTruth = x( 1, :);
x2_groundTruth = x( 2, :);

p = [ 0.025, 0.50, 0.975];

tArea = 1:N;
tArea = [ tArea, fliplr( tArea )];

% confidence intervals
x1_CI = quantile( X1, p);
x2_CI = quantile( X2, p);

x1_median = x1_CI( 2, :);
x1_lowerBound = x1_CI( 1, :);
x1_upperBound = x1_CI( 3, :);

x1Area = [ x1_lowerBound, fliplr( x1_upperBound )];

x2_median = x2_CI( 2, :);
x2_lowerBound = x2_CI( 1, :);
x2_upperBound = x2_CI( 3, :);

x2Area = [ x2_lowerBound, fliplr( x2_upperBound )];

figure();
tiledlayout("flow");

nexttile();
hold on

h = plot( x1_groundTruth );
h.LineWidth = 2;

h = plot( x1_median );
h.LineWidth = 2;

hFill = fill( tArea, x1Area, h.Color);
hFill.FaceAlpha = 0.25;
hFill.LineStyle = "none";

nexttile();
hold on

h = plot( x2_groundTruth );
h.LineWidth = 2;

h = plot( x2_median );
h.LineWidth = 2;

hFill = fill( tArea, x2Area, h.Color);
hFill.FaceAlpha = 0.25;
hFill.LineStyle = "none";
% end
% figure(2)
% plot(10:100:10000,els_t(1:))
% xlabel("Number of Particles")
% ylabel("Time")

%% Function
function [ indx ] = resampleSystematic( w )
N = length(w);
Q = cumsum(w);
T = linspace(0,1-1/N,N) + rand(1)/N;
T(N+1) = 1;
i=1;
j=1;
while (i<=N),
    if (T(i)<Q(j)),
        indx(i)=j;
        i=i+1;
    else
        j=j+1;
    end
end
end



