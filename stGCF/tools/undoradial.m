% undoradial    remove radial distortion
%
% [xl] = undoradial(x,K,kc)
%
% x ... 3xN coordinates of the distorted pixel points
% K ... 3x3 camera calibration matrix
% kc ... 4x1 vector of distortion parameters
% alpha_c ... scalar, skew distortion parameter (default value: zero)
%
% xl ... linearized pixel coordinates
%        these coordinates should obey the linear pinhole model
% 
% It calls comp_distortion_oulu: undistort pixel coordinates.
% function taken from the CalTech camera calibration toolbox

function [xl] = undoradial(x_kk,K,kc,alpha_c)

  if size(x_kk,1) ~= 3
    error( 'undoradial needs 3xN "xl" matrix of 2D points' );
  end
  
  if ~exist( 'alpha_c', 'var' )
    alpha_c = 0;
  end
  
  cc(1) = K(1,3);
  cc(2) = K(2,3);
  fc(1) = K(1,1);
  fc(2) = K(2,2);

  % First: Subtract principal point, and divide by the focal length:
  x_distort = [(x_kk(1,:) - cc(1))/fc(1); (x_kk(2,:) - cc(2))/fc(2)];

  % Second: compensate for skew
  x_distort( 1,: ) = x_distort( 1,: ) - alpha_c * x_distort( 2,: );
  
  if norm(kc) ~= 0,
    % Third: Compensate for lens distortion:
    xn = comp_distortion_oulu(x_distort,kc);
  else
    xn = x_distort;
  end;

  % back to the linear pixel coordinates
  xl = K*[xn;ones(size(xn(1,:)))];