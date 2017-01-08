import numpy
from context import svm, mlp
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

path = join('/', 'home', 'fabio', 'imagens_clodoaldo', 'Wavelet')
level = '3'
wavelet = 'sym5'
band = 'LL'
band_dict = {'LL':0, 'HL':1, 'LH':2, 'HH':3}
path = join(path, wavelet)
files = [f for f in listdir(path) if isfile(join(path, f))]
files = sorted(files, key=lambda file: int(file.split('.')[0][1:]))
X, y = [], []
i = 4
begin = i * 840
end = begin + 840
for file in files[begin:end]:
    try:
        dataset = loadmat(join(path, file))
    except Exception:
        continue
    finally:
        data = numpy.ravel(dataset['coef'][0][0][band_dict[band]][0,int(level)-1])
        X.append(data)
        y.append(int(file.split('.')[0][1:]))


clf1 = joblib.load('trained-estimators/'+wavelet+'-3-LL/svm-linear-'+str(i+1)+'.pkl')
clf2 = joblib.load('trained-estimators/'+wavelet+'-3-LL/svm-rbf-'+str(i+1)+'.pkl')
clf3 = joblib.load('trained-estimators/'+wavelet+'-3-LL/mlp-'+str(i+1)+'.pkl')

X, y = numpy.array(X), numpy.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
pca = PCA(n_components=0.9)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)

print(classification_report(y_test, clf1.predict(X_test)))
print(classification_report(y_test, clf2.predict(X_test)))
print(classification_report(y_test, clf3.predict(X_test_PCA)))
