import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import data

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)  
        self.model.fit(X, Y_)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)
    
    @property
    def support(self):
        return self.model.support_

def svm_decfun(model):
    def classify(X):
        return model.predict(X)
    return classify

if __name__ == "__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    svm_model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')

    Y = svm_model.predict(X)
    M, accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(f"Ukupna tocnost: {accuracy}, Odziv: {recall}, Preciznost: {precision}")

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decfun = svm_decfun(svm_model)
    data.graph_surface(decfun, bbox, offset=0.5)
    sv_indices = svm_model.support
    data.graph_data(X, Y_, Y, special=sv_indices)
    plt.show()
