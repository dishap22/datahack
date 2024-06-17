import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Reading in data
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')

train_data = pd.merge(train_features, train_labels, on='respondent_id')

x_train = train_data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y_train_xyz = train_data['xyz_vaccine']
y_train_seasonal = train_data['seasonal_vaccine']
x_test = test_features.drop(columns=['respondent_id'])


# Processing missing values based on if they're numerical or not
numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

x_train_preprocessed = preprocessor.fit_transform(x_train)
x_test_preprocessed = preprocessor.transform(x_test)


# Training models
model_xyz = LogisticRegression()
model_seasonal = LogisticRegression()

model_xyz.fit(x_train_preprocessed, y_train_xyz)
model_seasonal.fit(x_train_preprocessed, y_train_seasonal)


# Evaluation on training data
train_pred_xyz = model_xyz.predict_proba(x_train_preprocessed)[:, 1]
train_pred_seasonal = model_seasonal.predict_proba(x_train_preprocessed)[:, 1]

roc_auc_xyz = roc_auc_score(y_train_xyz, train_pred_xyz)
roc_auc_seasonal = roc_auc_score(y_train_seasonal, train_pred_seasonal)
mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2

print(f'ROC AUC for xyz_vaccine: {roc_auc_xyz}')
print(f'ROC AUC for seasonal_vaccine: {roc_auc_seasonal}')
print(f'Mean ROC AUC: {mean_roc_auc}')


# Predict and save test set values
test_pred_xyz = model_xyz.predict_proba(x_test_preprocessed)[:, 1]
test_pred_seasonal = model_seasonal.predict_proba(x_test_preprocessed)[:, 1]

submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': test_pred_xyz,
    'seasonal_vaccine': test_pred_seasonal
})

submission.to_csv('submission.csv', index=False)