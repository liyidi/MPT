function rep3d = p2dtop3d(p2d,z,cam, align_mat)
renew_x = undoradial(p2d,cam.K, [cam.kc 0], cam.alpha_c);
renew_xz = renew_x*z;
rexyz = [inv(cam.Pmat(1:3,1:3))* (renew_xz-cam.Pmat(:,4));1];
rep3d = align_mat*rexyz;
end