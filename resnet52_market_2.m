function net = resnet52_market_2()

% add?   add!
netStruct = load('./data/imagenet-resnet-50-dag.mat') ;
%netStruct = netStruct.net;
net = dagnn.DagNN.loadobj(netStruct) ;
net.removeLayer('fc1000');
net.removeLayer('prob');
net.removeLayer('pool5');
net.addLayer('GlobalPooling',dagnn.Global_Pooling,{'res5cx'},{'pool5'});

%---------setting1
for i = 1:numel(net.params)
    if(mod(i,2)==0)
        net.params(i).learningRate=0.02;
    else net.params(i).learningRate=0.001;
    end
    %name = net.params(i).name;
    %if(name(1)=='b')
     %   net.params(i).weightDecay=0;
    %end
end

%---
net.params(1).learningRate = 0.0001;


fc1Block = dagnn.Conv('size',[1 1 2048 512],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc1',fc1Block,{'pool5'},{'fc1'},{'fc1f','fc1b'});
net.addLayer('fc1bn',dagnn.BatchNorm(),{'fc1'},{'fc1bn'},...
    {'fc1bn_w','fc1bn_b','fc1bn_m'});
net.addLayer('relu_fc1bn',dagnn.ReLU('leak',0.1),{'fc1bn'},{'fc1bnx'});

dropoutBlock = dagnn.DropOut('rate',0.75);
net.addLayer('dropout',dropoutBlock,{'fc1bnx'},{'fc1d'},{});

fc751Block = dagnn.Conv('size',[1 1 512 751],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc751',fc751Block,{'fc1d'},{'prediction'},{'fc751f','fc751b'});
net.addLayer('softmaxloss',dagnn.Loss('loss','softmaxlog'),{'prediction','label'},'objective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;
net.initParams();

%net.conserveMemory = false;
%net.eval({'data',single(rand(256,128,3))});
end

