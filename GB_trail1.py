from sklearn.ensemble import GradientBoostingRegressor
from utils import load_data

import matplotlib.pyplot as plt
import numpy as np


input_feat = "base"
path_train = "Data/train_1.h5"
train_X, train_y = load_data(path_train, input_feat)
print(train_y.shape)
print(train_X.shape)

path_test = "Data/test_1.h5"
test_X, test_y = load_data(path_test, input_feat)
print(test_y.shape)
print(test_X.shape)
boosting=GradientBoostingRegressor(max_depth=5)
boosting.fit(train_X,train_y)
y_train_pred=boosting.predict(train_X)
y_test_pred=boosting.predict(test_X)
importance = boosting.feature_importances_
indices = np.argsort(importance)[::-1]
print(indices)
feature_name = np.array(['cement', 'water', 'Coarse aggregate', 'Fine aggregate', 'Super-plasticizer', 'Slag', 'Fly ash','Curing time'])
for f in range(train_X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 60,
                            feature_name[indices[f]],
                            importance[indices[f]]))

plt.figure()
plt.title('Feature Importance')
plt.bar(range(train_X.shape[1]),
        importance[indices],
        align='center')
plt.xticks(range(train_X.shape[1]),
           feature_name[indices], rotation=70)
plt.xlim([-1, train_X.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()
from sklearn.metrics import mean_squared_error,r2_score

print("MSE Train: %.3f, Test: %.3f" % (mean_squared_error(train_y,y_train_pred),
                                       mean_squared_error(test_y,y_test_pred)))
print("R2_Score Train: %.3f, Test: %.3f" % (r2_score(train_y,y_train_pred),
                                            r2_score(test_y,y_test_pred)))


plt.scatter(y_train_pred,train_y,label='Training Data')
plt.scatter(y_test_pred, test_y,label='Test Data')
plt.xlabel("Predicted Values (MPa)")
plt.ylabel("Real Values (MPa)")
plt.legend(loc='upper left')
#plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='black')
#plt.xlim([-10,50])
#plt.savefig('./fig1.png')
plt.show()