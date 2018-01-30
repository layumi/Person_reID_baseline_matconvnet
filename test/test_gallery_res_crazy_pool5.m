% In this file, we densely extract the feature
% It's similar to the 10-crop in the ResNet Paper.
clear;
netStruct = load('../data/res52_baseline_batch8_fc512_all_small_init/net-epoch-40.mat');
%--------add norm
net = dagnn.DagNN.loadobj(netStruct.net);
net.addLayer('lrn_test',dagnn.LRN('param',[4096,0,1,0.5]),{'pool5'},{'pool5n'},{});
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.conserveMemory = true;
im_mean = net.meta(1).normalization.averageImage;
im_mean = mean(mean(im_mean,1),2);
p = dir('/home/zzd/market1501/gallery_pytorch/bounding_box_test/*jpg');
ff = [];
%%------------------------------

for i = 1:16:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(16,numel(p)-i+1)
        str = strcat('/home/zzd/market1501/gallery_pytorch/bounding_box_test/',p(i+j-1).name);
        imt = imresize(imread(str),[288,144]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save('../test/resnet_gallery_pool5n.mat','ff','-v7.3');
%}

%---------query
p = dir('/home/zzd/market1501/query_pytorch/query/*jpg');
ff = [];
for i = 1:16:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(16,numel(p)-i+1)
        str = strcat('/home/zzd/market1501/query_pytorch/query/',p(i+j-1).name);
        imt = imresize(imread(str),[288,144]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save('../test/resnet_query_pool5n.mat','ff','-v7.3');
