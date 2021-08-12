# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import csv
from tqdm import tqdm

from scipy.optimize import minimize


# %%
## Read the data
data = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

data['song_id'] = data['song_id'] - 1
test['song_id'] = test['song_id'] - 1

## Encode the user ids
le = preprocessing.LabelEncoder()
le.fit(data['customer_id'].unique())
data['customer_id'] = le.transform(data['customer_id'])
test['customer_id'] = le.transform(test['customer_id'])

## Train-val split
train, val = train_test_split(data, test_size=0.2, random_state=2)

print("Train No: customers", len(train['customer_id'].unique()))
print("Train No: songs", len(train['song_id'].unique()))

print("No: customers", len(data['customer_id'].unique()))
print("No: songs", len(data['song_id'].unique()))

assert len(train['customer_id'].unique()) == len(data['customer_id'].unique())
assert len(train['song_id'].unique()) == len(data['song_id'].unique())


# %%
def compute_mse(ds):
    '''
        Helper function to compute mse from CF model
    '''
    y_true = []
    y_pred = []
    for ind in ds.index:
        uid = ds['customer_id'][ind]
        iid = ds['song_id'][ind]
        true = ds['score'][ind]
        pred = model.predict(uid, iid)[0]
        if pred>=5:
            pred = 5
        y_true.append(true)
        y_pred.append(pred)
    mse = mean_squared_error(y_true, y_pred)
    return mse  


# %%
class CF_Regressor(object):

    def __init__(self, n_features = 2, lamda = 0.0, max_iters = 400, random_seed = 0):

        self._n_features = n_features
        self._lamda = lamda
        self._max_iter = max_iters
        self._reg_flag = True if self._lamda > 0 else False

        np.random.seed(1)

    def _init_weights(self, n_iids, n_uids):
        U = np.random.rand(n_uids, self._n_features) # User embedding matrix
        V = np.random.rand(n_iids, self._n_features) #  Item embedding matrix
        self._weights = np.concatenate((U.flatten(), V.flatten()))

    def _get_weights(self, weights):
        U = weights[:self.n_uids * self._n_features].reshape(self.n_uids, self._n_features)
        V = weights[self.n_uids * self._n_features:].reshape(self.n_iids, self._n_features)
        return U, V

    def _grad(self, weights, A, R):
        U, V = self._get_weights(weights)
 
        preds = np.matmul(U, V.T)
        preds_err = (preds - A) * R
                
        U_grad = np.matmul(preds_err, V)
        V_grad = np.matmul(preds_err.T, U)

        if self._reg_flag > 0:
            V_grad += V * self._lamda
            U_grad += U * self._lamda

        return np.concatenate((U_grad.flatten(), V_grad.flatten()))

    def _cost(self, weights, A, R):
        cost = 0
        
        ## MSE
        U, V = self._get_weights(weights)
        preds = np.matmul(U, V.T)
        mse = np.power(preds - A, 2)
        mse = 0.5 * np.sum(mse * R)
        cost += mse
        
        ## Regularization
        if self._reg_flag:
            reg_cost = 0.5 * self._lamda * np.sum(np.power(weights, 2))
            cost += reg_cost
 
        return cost

    def fit(self, A, R):
        '''
            A: Ratings matrix
            R: matrix of score present (0 or 1)
        '''
        self._A = A
        self._R = R

        # Init weights
        self.n_uids, self.n_iids = A.shape
        self._init_weights(self.n_iids, self.n_uids)
        
        self._means = np.nanmean(np.where(R, A, np.nan), axis=1, keepdims=True)
        centered_A = np.where(R, A - self._means, 0)

        # Minimize
        optimized = minimize(fun=self._cost, x0=self._weights, args=(centered_A, R), method='TNC', jac=self._grad, options={'maxiter':self._max_iter})
        self._weights = optimized.x

    def predict(self, uid, iid):
        '''
        uid - User ID (0 index)
        iid - Item/Song ID (0 index)
        '''
        U, V = self._get_weights(self._weights)
        return np.matmul(U[uid, :], V[iid, :].T) + self._means[uid]


# %%
## Training
mat = np.zeros((14053, 10000))
R_mat = np.zeros((14053, 10000))

for i in train.index:
    u_id = train['customer_id'][i]
    i_id = train['song_id'][i]
    mat[u_id, i_id] = train['score'][i]
    R_mat[u_id, i_id] = 1
    
params = {
    "n_features" : 2,
    "lamda" : 3,
    "max_iters" : 300, #200
}

model = CF_Regressor(**params)
model.fit(mat, R_mat)

train_mse = compute_mse(train)
val_mse = compute_mse(val)
print("train_mse={}, val_mse={}".format(train_mse, val_mse))
print("\n")

## Training all
mat = np.zeros((14053, 10000))
R_mat = np.zeros((14053, 10000))

for i in data.index:
    u_id = data['customer_id'][i]
    i_id = data['song_id'][i]
    mat[u_id, i_id] = data['score'][i]
    R_mat[u_id, i_id] = 1
    
model = CF_Regressor(**params)
model.fit(mat, R_mat)

train_mse = compute_mse(train)
val_mse = compute_mse(val)
print("train_mse={}, val_mse={}".format(train_mse, val_mse))
print("\n")


# %%
f = open('submission12.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(["test_row_id", "score"])

for ind in test.index:
    customer_id = test['customer_id'][ind]
    song_id = test['song_id'][ind]
    pred = model.predict(customer_id, song_id)[0]
    if pred>=5:
        pred = 5
    writer.writerow([ind, pred])
f.close()


