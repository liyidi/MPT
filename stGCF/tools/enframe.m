function f=enframe(x,win,inc)            %把语音信号按帧长和帧移进行分帧
%ENFRAME split signal up into (overlapping) frames: one per row. F=(X,WIN,INC)
%
%	F = ENFRAME(X,LEN) splits the vector X(:) up into
%	frames. Each frame is of length LEN and occupies
%	one row of the output matrix. The last few frames of X
%	will be ignored if its length is not divisible by LEN.
%	It is an error if X is shorter than LEN.
%
%	F = ENFRAME(X,LEN,INC) has frames beginning at increments of INC
%	The centre of frame I is X((I-1)*INC+(LEN+1)/2) for I=1,2,...
%	The number of frames is fix((length(X)-LEN+INC)/INC)
%
%	F = ENFRAME(X,WINDOW) or ENFRAME(X,WINDOW,INC) multiplies
%	each frame by WINDOW(:)

%	   Copyright (C) Mike Brookes 1997
%      Version: $Id: enframe.m,v 1.4 2006/06/22 19:07:50 dmb Exp $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nx=length(x(:));                              %取数据长度
nwin=length(win);                             %取窗长
if (nwin == 1)                                %窗长是否为1，为1表示没有设窗函数
   len = win;                                 %是，帧长=win
else     
   len = nwin;                                %否，帧长=窗长
end
if (nargin < 3)                               %如果只有两个参数，帧移=帧长
   inc = len;
end
nf = fix((nx-len+inc)/inc);                   %计算帧数
f=zeros(nf,len);                              %初始化
indf= inc*(0:(nf-1)).';                       %每帧在x中开始的位置
inds = (1:len);                               %每帧的数据对应1：len
f(:) = x(indf(:,ones(1,len))+inds(ones(nf,1),:));%把数据分帧
if (nwin > 1)                                 %若参数中包括窗函数，把每帧乘以窗函数
    w = win(:)';
    f = f .* w(ones(nf,1),:);                %f为分帧后的数组，帧数*帧长
end


