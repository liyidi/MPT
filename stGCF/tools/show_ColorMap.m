function show_ColorMap(data)
%data is a 2d matrix,����Ϊ���󣬲����Ǳ�����ʸ����
figure()
surf(data)
shading interp
set(gca,'YDir','reverse');%y�ᷴ������ͼƬ����ϵһ�£�ʹԭ�������Ͻǡ�
view([0,90])
end