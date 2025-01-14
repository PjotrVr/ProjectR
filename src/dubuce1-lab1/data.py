import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    """
    Random bivariate normal distribution sampler.

    Hardwired parameters:
        horizontal_min, horizontal_max: horizontal range for the mean
        vertical_min, vertical_max: vertical range for the mean
        scale: controls the covariance range

    Methods:
        __init__: creates a new distribution
        get_sample(n): samples n datapoints
    """

    horizontal_min = 0 
    horizontal_max = 10
    vertical_min = 0 
    vertical_max = 10
    scale = 5

    def __init__(self):
        horizontal_range, vertical_range = self.horizontal_max - self.horizontal_min, self.vertical_max - self.vertical_min
        mean = (self.horizontal_min, self.vertical_min)
        mean += np.random.random_sample(2) * (horizontal_range, vertical_range)

        # Variances for principle directions (horizontal/vertical)
        eigvals = np.random.random_sample(2)
        eigvals *= (horizontal_range / self.scale, vertical_range / self.scale)
        eigvals **= 2

        # Pick random rotation [0, 1> * 2pi = [0, 2pi>
        theta = np.random.random_sample() * np.pi * 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)]
        ])

        # Covariance matrix
        Sigma = R.T @ np.diag(eigvals) @ R

        self.get_sample = lambda n: np.random.multivariate_normal(mean, Sigma, n)
    
# One Gaussian source per class
def sample_gauss_2d(num_classes, num_samples_per_class):
    # Create Gaussians
    Gs, ys = [], []
    for i in range(num_classes):
        Gs.append(Random2DGaussian())
        ys.append(i)

    # Sample dataset
    X = np.vstack([G.get_sample(num_samples_per_class) for G in Gs])
    y = np.hstack([[y] * num_samples_per_class for y in ys])
    return X, y

# One class can have multiple Gaussian components
def sample_gmm_2d(num_components, num_classes, num_samples_per_class):
    # Create Gaussian components and assign them random class idx
    Gs, ys = [], []
    for _ in range(num_components):
        Gs.append(Random2DGaussian())
        ys.append(np.random.randint(num_classes))

    # Sample dataset
    X = np.vstack([G.get_sample(num_samples_per_class) for G in Gs])
    y = np.hstack([[y] * num_samples_per_class for y in ys])
    return X, y


def graph_surface(function, rect, offset=0.5, width=256, height=256):
  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  #get the values and reshape them
  values=function(grid).reshape((width,height))
  
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval, cmap="jet")

  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])

def graph_data(X,Y_, Y, special=[]):
  # colors of the datapoint markers
  palette=([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
  colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
  for i in range(len(palette)):
    colors[Y_ == i] = palette[i]

  # sizes of the datapoint markers
  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  
  # draw the correctly classified datapoints
  good = (Y_ == Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], 
              s=sizes[good], marker='o', edgecolors='black')

  # draw the incorrectly classified datapoints
  bad = (Y_ != Y)
  plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad], s=sizes[bad], marker='s', edgecolors='black')

def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh

def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    fn = sum(np.logical_and(Y != Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp+fn + tn+fp)
    return accuracy, recall, precision

  
def eval_perf_multi(Y, Y_):
    """
    Evaluate performance of multiclass classification

    Parameters
    ----------
        Y: predicted class indices, np.array of shape Nx1
        Y_: ground truth indices, np.array of shape Nx1

    Returns
    -------
        M: confusion_matrix
        accuracy, precision, recall: metrics

    """
    
    Y_ = Y_.reshape(-1)
    Y = Y.reshape(-1)
    num_classes = np.max(Y_) + 1
    M = np.zeros((num_classes, num_classes))
    for y_, y in zip(Y_, Y):
        M[y_][y] += 1

    accuracy = np.trace(M) / np.sum(M)
    precision = np.divide(np.diag(M), np.sum(M, axis=0), where=np.sum(M, axis=0) != 0)
    recall = np.divide(np.diag(M), np.sum(M, axis=1), where=np.sum(M, axis=1) != 0)

    return M.astype(np.int32), accuracy, recall, precision

def eval_AP(ranked_labels):
  """Recovers AP from ranked labels"""
  
  n = len(ranked_labels)
  pos = sum(ranked_labels)
  neg = n - pos
  
  tp = pos
  tn = 0
  fn = 0
  fp = neg
  
  sumprec=0
  #IPython.embed()
  for x in ranked_labels:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)    

    if x:
      sumprec += precision
      
    #print (x, tp,tn,fp,fn, precision, recall, sumprec)
    #IPython.embed()

    tp -= x
    fn += x
    fp -= not x
    tn += not x

  return sumprec/pos

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__ == "__main__":
    np.random.seed(100)

    # Generate data
    X, y_true = sample_gmm_2d(4, 2, 30)
    #X, y_true = sample_gauss_2d(2, 100)

    # Predict Y
    y_pred = myDummyDecision(X) > 0.5 

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, rect, offset=0)
    graph_data(X, y_true, y_pred, special=[])
    plt.show()