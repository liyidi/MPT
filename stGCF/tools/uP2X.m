% uP2X ... linear reconstruction of 3D points
%          from N-perspective views
% 
% X = uP2X(Umat,Pmat); 
% Umat ... 3*N x n matrix of n homogenous points
% Ps ...   3*N x 4 matrix of projection matrices
% 
% X ... 4 x n matrix of homogenous 3D points
%
%  Watch out... slight difference with original code in the format of Ps
% 
% Algorithm is based on: Hartley and Zisserman, Multiple
% View Geometry, 2000, pp 297-298

function X = uP2X(Umat,Pmat); 

N = size(Umat,1)/3;
n =	size(Umat,2);

% reshuffle the Ps matrix
%Pmat = [];
%for i=1:N;
%  Pmat = [Pmat;Ps(:,i*4-3:i*4)];
%end

X = [];
for i=1:n,	% for all points
  A = [];	
  for j=1:N,	% for all cameras
	% create the data matrix
	A = [A; Umat(j*3-2,i)*Pmat(j*3,:) - Pmat(j*3-2,:); Umat(j*3-1,i)*Pmat(j*3,:) - Pmat(j*3-1,:)];
  end
  [u,s,v] = svd(A);
  X = [X,v(:,end)];
end
% normalize reconstructed points
X = X./repmat(X(4,:),4,1);

return;