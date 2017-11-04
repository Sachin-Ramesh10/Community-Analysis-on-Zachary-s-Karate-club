clear all;
fileName = 'karate.gml';
inputfile = fopen(fileName);
A=[];

l=0;
k=1;
while 1

    % Get a line from the input file
    tline = fgetl(inputfile);

    % Quit if end of file
    if ~ischar(tline)
        break
    end
    
    nums = regexp(tline,'\d+','match'); %get number from string
    if length(nums)
        if l==1
            l=0;
            A(k,2)=str2num(nums{1}); 
            k=k+1;
            continue;
        end
        A(k,1)=str2num(nums{1});
        l=1;
    else
        l=0;
        continue;
    end 
end
disp('Processing Edge-List File')

A1 = A;

dim = max(max(A1));
[E_Size, junk] = size(A1);

sprintf('The dataset has %d nodes and %d edges',dim, E_Size)

disp('Filling Adjanceny Matrix')

adj2 = sparse(A(:,1), A(:,2), ones(E_Size,1), dim, dim, E_Size);

Communities = GCSpectralClust1(adj2,2);


function W=PermMat(N)

W=zeros(N,N);
q=randperm(N);
for n=1:N
	W(q(n),n)=1; 
end
end

function VV=GCSpectralClust1(A,Kmax)

N=length(A);
W=PermMat(N);                     
A=W*A*W';

VV(:,1)=ones(N,1);
for k=2:Kmax
	[ndx,Pi,cost]= grPartition(A,k,1);
	VV(:,k)=ndx;
end

VV=W'*VV;                        

end



function [ndx,Pi,cost]= grPartition(C,k,nrep)

if nargin<3
  nrep=1;
end

[n,m]=size(C);
if n~=m
  error('grPartition: Cost matrix is not square'); 
end  

if ~issparse(C)
  C=sparse(C);  
end

% Test for symmetry
if any(any(C~=C'))
  
  C=(C+C')/2;
end  

% Test for double stochasticity
if any(sum(C,1)~=1)

  C=C/(1.001*max(sum(C)));  % make largest sum a little smaller
                            % than 1 to make sure no entry of C becomes negative
  C=C+sparse(1:n,1:n,1-sum(C));
  if any(C(:))<0
    error('grPartition: Normalization resulted in negative costs. BUG.')
  end
end  

if any(any(C<0))
  error('grPartition: Edge costs cannot be negative')
end  

% Spectral partition
options.issym=1;               % matrix is symmetric
options.isreal=1;              % matrix is real
options.tol=1e-6;              % decrease tolerance 
options.maxit=500;             % increase maximum number of iterations
options.disp=0;
[U,D]=eigs(C,k,'la',options);  % only compute 'k' largest eigenvalues/vectors


   ndx=mykmeans1(U,k,100,nrep);

if nargout>1
  Pi=sparse(1:length(ndx),ndx,1);
end  

if nargout>2
  cost=full(sum(sum(C))-trace(Pi'*C*Pi));
end

end

function bestNdx=mykmeans1(X,k,nReplicates,maxIterations)


if nargin<4,
  maxIterations=100;
end
if nargin<3,
  nReplicates=1;  
end
    
nPoints=size(X,1);
if nPoints<=k, 
  bestNdx=1:nPoints;
  return
end

% normalize vectors so that inner product is a distance measure
normX=sqrt(sum(X.^2,2));
X=X./(normX*ones(1,size(X,2)));
bestInnerProd=0; % best distance so far

for rep=1:nReplicates

  % random sample for the centroids
  ndx = randperm(size(X,1));
  centroids=X(ndx(1:k),:);
  
  lastNdx=zeros(nPoints,1);
  
  for iter=1:maxIterations
    InnerProd=X*centroids'; % use inner product as distance
    [maxInnerProd,ndx]=max(InnerProd,[],2);  % find 
    if ndx==lastNdx,
      break;          % stop the iteration
    else
      lastNdx=ndx;      
    end
    for i=1:k
      j=find(ndx==i);
      if isempty(j)     
	%error('mykmeans: empty cluster')
      end
      centroids(i,:)=mean(X(j,:),1);
      centroids(i,:)=centroids(i,:)/norm(centroids(i,:)); % normalize centroids
    end
  end
  if sum(maxInnerProd)>bestInnerProd
    bestNdx=ndx;
    bestInnerProd=sum(maxInnerProd);
  end

end % for rep
 
end