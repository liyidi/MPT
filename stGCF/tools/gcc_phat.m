function [R, t] = gcc_phat(x1, x2)

leng = min(length(x1),length(x2));
x1 = reshape(x1,1,numel(x1));
x2 = reshape(x2,1,numel(x2));
x1 = [x1,zeros(1,length(x1))];
x2 = [zeros(1,length(x2)),x2];
X1=fft(x1);
X2=fft(x2);
Pxy=X1.*conj(X2)./(abs(X1).*abs(X2)+eps); 
R=ifft(Pxy);
R=real(R);
R = abs(R);
t = 1-leng:length(R)-leng;

end