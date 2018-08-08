addpath(genpath('../../RMSG'))

%% Init parameters
k=4;

% RST=0; % do not start from scratch, continue on previous run!
RST=1; % start from scratch, do not continue on previous run!

numiters=5; maxiter=Inf;

%% for mnist
eta=2;

methods={'batch','incremental','msg','l1rmsg','l2rmsg','l21rmsg'};

%% Sample from the orthogonal distribution
% d=32; N=10000; tau=1.3; dataname=sprintf('orthogonal_N=%d_d=%d_tau=%g',N,d,tau);
% [X,~]=simdist(d,3*N,tau);
% data=struct('training',X(:,1:N),'tuning',X(:,N+1:2*N),...
%     'testing',X(:,2*N+1:3*N));

%% Arbitrary Dataset
% load('mnist_pca.mat'); dataname='mnist_pca'; tau=0;
syn_gen; load('syn_exp_decay.mat'); dataname='syn_exp_decay'; tau=0;

%% normalize the dataset
N=size(data.training,2);
N=min(N,maxiter);
data.training=data.training(:,1:N);
mu=mean(data.training,2);
data.training=data.training-repmat(mu,1,N);
data.tuning=data.tuning-repmat(mu,1,size(data.tuning,2));
data.testing=data.testing-repmat(mu,1,size(data.testing,2));


%% just for demonsteration, in practice we cross-validate lambda, beta and eta
Cx=data.training*data.training'/N;
L=eig(Cx); L=sort(L,'descend');
lambda=L(k)-L(k+1);
beta=(L(k)+L(k+1))/4;

%% run the algorithms
stochPCA(dataname,methods,data,k,numiters,eta,lambda,RST,beta);

%% plot the results
plotobjV(dataname,methods,k,N,numiters,eta,lambda,beta);
