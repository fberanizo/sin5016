% bands: CA, CH, CV, CD
% decomposition levels: 1, 2, 3, 4
X=[];
y={};
files=dir('/home/fabio/imagens_clodoaldo/Wavelet/db2/*.mat');
for i=1:length(files)
   load(strcat('/home/fabio/imagens_clodoaldo/Wavelet/db2/',files(i).name));
   LL=reshape(coef.CA{3},[1,numel(coef.CA{3})]);
   X=[X;reshape(LL,[1,numel(LL)])];
   tokens=regexp(files(i).name, 'p([0-9]+)', 'tokens');
   y={y;tokens{1}};
end
clf=SVM('linear');
clf.fit(X,y);