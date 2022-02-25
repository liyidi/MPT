%%%-----------griddata to image size-------------------------
[X,Y] = meshgrid(1:Data.obj.Width,1:Data.obj.Height);
for fr = 1:size(gcfmap_max,1)
    for c = 1:size(gcfmap_max{1,1},2)
        z = gcfmap_max{fr,1}(:,c);
        vq = griddata(sample2d(:,1),sample2d(:,2),z,X,Y,'v4');
        gcfmap_grid{fr,1}(:,:,c) = vq;
    fprintf(['griddata for cut: ' num2str(c) '\n'])    
    end
    fprintf(['convert gcfmap of frame: ' num2str(fr) '\n'])
end