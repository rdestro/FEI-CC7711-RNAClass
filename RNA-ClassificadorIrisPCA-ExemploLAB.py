
from sklearn.datasets import load_iris

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = load_iris()

features =data.data
target = data.target

neuronios = (10,10,10)
epocas = 5000

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.title("Feature originais ()")
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis')

Classificador = MLPClassifier(hidden_layer_sizes = neuronios, alpha=1, max_iter=epocas, n_iter_no_change=epocas)
Classificador.fit(features,target)
predicao = Classificador.predict(features)

plt.subplot(2,2,3)
plt.title("Classificação sem PCA")
plt.scatter(features[:,0], features[:,1], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis',s=15)




pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
pca = pca.fit(features)
pca_features = pca.transform(features)
print('Mantida %5.2f%% da informação do conjunto inicial de dados'%(sum(pca.explained_variance_ratio_)*100))

plt.subplot(2,2,2)
plt.title("Feature com PC (2 componentes)")
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis')


ClassificadorPCA = MLPClassifier(hidden_layer_sizes = neuronios, alpha=1, max_iter=epocas, n_iter_no_change=epocas)
ClassificadorPCA.fit(pca_features,target)


pca_predicao = ClassificadorPCA.predict(pca_features)

plt.subplot(2,2,4)
plt.title("Classificação com PCA")
plt.scatter(pca_features[:,0], pca_features[:,1], c=pca_predicao,marker='d',cmap='viridis',s=150)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis',s=15)
#plt.show()


cm = confusion_matrix(target, predicao)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=data.target_names)
disp.plot()
plt.show()


cm_pca = confusion_matrix(target, pca_predicao)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pca,display_labels=data.target_names)
disp.plot()
plt.show()


