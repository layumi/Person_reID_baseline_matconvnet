clear;
p = '/home/zzd/market1501/bounding_box_train/';
pp = '/home/zzd/market1501/bounding_box_train/*.jpg';
pw = '/home/zzd/market1501/bounding_box_train_288_144/';
mkdir(pw);
imdb.meta.sets=['train','test'];
file = dir(pp);
counter_data=1;
counter_last = 1;
class = 0;
c_last = '';
cc = [];
im_mean = [];
for i=1:length(file)
    url = strcat(p,file(i).name);
    im = imread(url);
    im = imresize(im,[288,144]); % save 256*256 image in advance for faster IO
    im_mean = cat(4,im_mean,mean(mean(im,1),2));
    url256 = strcat(pw,file(i).name);
    imwrite(im,url256);
    imdb.images.data(counter_data) = cellstr(url256); 
    c = strsplit(file(i).name,'_');
    if(~isequal(c{1},c_last))
        class = class + 1;
        fprintf('%d::%d\n',class,counter_data-counter_last);
        cc=[cc;counter_data-counter_last];
        c_last = c{1};
        counter_last = counter_data;
    end
    imdb.images.label(:,counter_data) = class;
    counter_data = counter_data + 1;
end
s = counter_data-1;
imdb.images.set = ones(1,s);
imdb.images.set(:,randi(s,[round(0.1*s),1])) = 2;

% no validation for small class 
cc = [cc;9];
cc = cc(2:end);
list = find(imdb.images.set==2);
for i=1:numel(list)
    if cc(imdb.images.label(list(i)))<10
        imdb.images.set(list(i))=1;
    end
end

mean(im_mean,4)
save('url_data_288x144.mat','imdb','-v7.3');
