function p3d = reconstruct_3d( p2d, cam, align_Pmat )
%输入的P2D格式是(row col 1)'
% Check parameters
  
  if nargin < 3
    error( 'reconstruct_3d: needs 3 paramaters' );
  end
  
  if ~isstruct( cam )
    error( [ 'reconstruct_3d: "cam" must be a structure array with fields ' ...
	   'Pmat, K, kc, alpha_c.' ] );
  end
%  ncams = length( cam );
   ncams =1;%只使用一个摄像头
%   siz = size( p2d );
%   if length( siz ) ~= 2
%     error( 'reconstruct_3d: "p2d" must be (3*ncams) x npoints matrix' );    
%   end
%   if siz( 1 ) ~= 3*ncams
%       error( 'reconstruct_3d: "p2d" must be (3*ncams) x npoints matrix' );
%   end
%   npoints = siz( 2 );
  
  siz = size( align_Pmat );
  if length( siz ) ~= 2
    error( 'reconstruct_3d: "align_Pmat" must be 4 x 4 matrix' );
  end
  if ~all( siz == [ 4 4 ] )
    error( 'reconstruct_3d: "align_Pmat" must be 4 x 4 matrix' );    
  end
  
  % Remove radial distortion
  x = p2d;
  for c = 1:ncams
    p2d_ind = (c*3-2):(c*3);
    x( p2d_ind, : ) = undoradial( x( p2d_ind, : ), cam( c ).K, [cam( c ).kc 0], cam( c ).alpha_c );
  end
  
  % Call uP2X to reconstruct 3D coordinates from 2d coordinates
  X = uP2X( x, cat( 1, cam.Pmat ) );
  
  % alignment with desired referent
  p3d = align_Pmat * X;