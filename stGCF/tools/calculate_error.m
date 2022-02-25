%[x,y,w,h]
for frmap = 1:size(gcfmap_max,1)
frgt = frmap+startGT-1;
GT3d = Data.GT3D(frgt,2:4);
gt3d = [GT3d 1]';
gt2d = project(gt3d,Data.cam,Data.align_mat) ;
gt2d = gt2d(cam_number,1:2);
for i = 1:Para.m2
    [maxval, maxindcut]=max(gcfmap_max{frmap,1}(:,i));
    loc2d= sample2d(maxindcut,:);
    error(i) = pdist2(loc2d,gt2d ,'Euclidean');
end
error2d(frmap) = mean(error);
end
