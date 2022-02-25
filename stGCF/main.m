clear all
%%%---initialize----
filepath = char({ '\AV163\'});
savepath = '\data\';
sequence= char({ 'seq12-1p-0100'});cam_number = 3;
Para.array_number = 3; % array_number = 1 or 2 or 3(1+2)

initialize_gcfmap;

w = 9:18:Data.obj.Width;
h = 9:18:Data.obj.Height;
sample2d=[];
for i = 1:size(w,2)
    for j = 1:size(h,2)
        sample2d = [sample2d;[w(i) h(j)]];
    end
end
%2d--3d
for i = 1:Para.map_num
    sample3dmap = [];
    z = Para.Blist(cam_number) + Para.Ilist(cam_number) * (i-1);%seq11cam1:0.5 + 0.5 * (i-1);seq11cam3:1.5 + 0.4 * (i-1);
    for s = 1: size(sample2d,1)
    p2d = [sample2d(s,:) 1]';
    p3d = p2dtop3d(p2d,z,Data.cam(cam_number), Data.align_mat);
    p3d= p3d(1:3,1)';
    sample3dmap = [sample3dmap;p3d];
    end
    sample3d(:,:,i) = sample3dmap;
end
%sample3D--¡·tau
for g = 1:Para.map_num
    tau3dlist = [];
    for i = 1:size(sample3d,1)
            p = sample3d(i,:,g);
            t = 1;
            for mici = 1:size(Data.audio,1)
                for micj = mici+1:size(Data.audio,1)
                    tauijk = (pdist2(p,Data.mic_posgt(mici,:),'Euclidean')...
                    -pdist2(p,Data.mic_posgt(micj,:),'Euclidean'))/Para.c;
                    tau3dlist(i,t)=round(tauijk*Para.fs);%x*y*z3D
                    t = t+1;
                end
            end
    tau3d{g,1} =     tau3dlist;%tau3d£ºs
    end
end

%%%-----gcfmap---
[startGT,endGT] = deal(Para.startGTlist(cam_number),Para.endGTlist(cam_number));
[startFR,endFR] = deal(Data.GT3D(startGT,cam_number+4),Data.GT3D(endGT,cam_number+4));
[startFA,endFA] = deal(2*startFR-3,2*endFR-3);%a = 2*v-3
gcfmap_v2 = [];
for fn = startFA-20:endFA
j = 1;
for mici = 1:size(Data.audio,1)
for micj = mici+1:size(Data.audio,1)
[tau,R,lag] = gccphat(Data.frame{mici,1}(fn,:)',Data.frame{micj,1}(fn,:)');%lag[-(N-1),(N-1)]
for g = 1:size(tau3d,1)
for i = 1:size(tau3d{1,1},1)
    if sample3d(i,1,g)>-1.8 && sample3d(i,1,g)<1.8 ...
            && sample3d(i,2,g)>-7.2 && sample3d(i,2,g)<2 ...
                && sample3d(i,3,g)>-0.04 && sample3d(i,3,g)<1.56                             
        indexij = N+tau3d{g,1}(i,j);
        Rp{g,1}(i,j) = R(indexij);
    else
        Rp{g,1}(i,j) = 0;
    end
end
end
j = j+1;
end
end
gcfmapcut= [];
for c = 1:size(tau3d,1)
    gcfmapcut(:,c)= real(sum(Rp{c,1},2));
end
gcfmap_fa{fn,1}=gcfmapcut;
fprintf(['get gcfmap of audio frame ' num2str(fn) '\n'])
end
gcfmap_fa_savename = ['gcfmaps_fa_seq45cam' num2str(cam_number) 'mic' num2str(Para.array_number) '.mat'];
%save([savepath gcfmap_fa_savename],'gcfmap_fa')

gcfdata = gcfmap_fa;
for frgt = startGT:endGT
FrameNumber = Data.GT3D(frgt,cam_number+4); 
fn = 2*FrameNumber-3;
[m,maxind] = sort(arrayfun(@(x) max(max(gcfdata{x,1})),fn-Para.m1:fn),'descend');
fnmax = fn-Para.m1 + maxind(1:Para.m2) -1;
for i = 1:Para.m2
    [m, maxindcol]=max(max(gcfdata{fnmax(i),1}));
    gcfmap_max{frgt-startGT+1,1}(:,i)=gcfdata{fnmax(i),1}(:,maxindcol);
end
fprintf(['get gcfmap_max of frame ' num2str(frgt) '\n'])
end
gcfmap_max_savename = ['gcfmaps_max_seq24cam' num2str(cam_number) 'mic' num2str(Para.array_number) '.mat'];
%save([savepath gcfmap_max_savename],'gcfmap_max')
grid_data;

calculate_error; % Error Calculation

figurize_gcfmap;  % plot figures



