function show_ColorMap(data)
%data is a 2d matrix,必须为矩阵，不能是标量或矢量。
figure()
surf(data)
shading interp
set(gca,'YDir','reverse');%y轴反方向，与图片坐标系一致，使原点在左上角。
view([0,90])
end