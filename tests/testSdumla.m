% bands: CA, CH, CV, CD
% decomposition levels: 1, 2, 3, 4
X=[];
files=dir('D:\Wavelet\db2\*.mat');
for i=1:length(files)
   load(strcat('D:\Wavelet\db2\',files(i).name));
   row=reshape(coef.CA{3},[1,numel(coef.CA{3})]);
   X=[X;reshape(LL,[1,numel(LL)])];
end
clf=SVM('linear')
clf.fit(X)