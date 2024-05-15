#Anya Kondamani -- ak8699 -- Capstone Project!!

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,classification_report,roc_auc_score, roc_curve, accuracy_score
from sklearn.decomposition import PCA 
from sklearn.preprocessing import LabelEncoder

random.seed(19884614)

spotify_data = "spotify52kData.csv"
data = pd.read_csv(spotify_data)

#Question 1
features=data[['duration','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']]
fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(30, 12))
axes = axes.flatten()
for i, feature in enumerate(features.columns):
    axes[i].hist(features[feature], bins=51, color='pink',edgecolor='black')
    axes[i].set_title('Histogram of {}'.format(feature))
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

#Question 2
#dp=duration_popularity
dp_pearson_r,dp_pearson_p_val=stats.pearsonr(data['duration'], data['popularity'])
dp_spearman_r,dp_spearman_p_val=stats.spearmanr(data['duration'], data['popularity'])

print('Pearson Coefficient: ',dp_pearson_r,"; P-value: ",dp_pearson_p_val)
print('Spearman Coefficient: ',dp_spearman_r,"; P-value: ",dp_spearman_p_val)

plt.scatter(data['duration'], data['popularity'],color='crimson')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.title('Duration vs. Popularity')
plt.show()

#Question 3
#x=explicitness
explicit=data[data['explicit']==True]['popularity']
not_explicit=data[data['explicit']==False]['popularity']
mean_pop_explicit = explicit.mean()
mean_pop_not_explicit = not_explicit.mean()

print('Average popularity of explicit songs = ',mean_pop_explicit)
print('Average popularity of non-explicit songs = ',mean_pop_not_explicit)

x_mwu_statistic, x_mwu_p_value = stats.mannwhitneyu(explicit, not_explicit)
print("Mann-Whitney U Test: U-Statistic = ",x_mwu_statistic," P-value = ",x_mwu_p_value)

#Question 4
major_key=data[data['mode']==1]['popularity']
minor_key=data[data['mode']==0]['popularity']

mean_pop_major = major_key.mean()
mean_pop_minor = minor_key.mean()

print('Average popularity of songs in major key = ',mean_pop_major)
print('Average popularity of songs in minor key = ',mean_pop_minor)

key_mwu_statistic, key_mwu_p_value = stats.mannwhitneyu(major_key, minor_key)
print("Mann-Whitney U Test: U-Statistic = ",key_mwu_statistic," P-value = ",key_mwu_p_value)

#Question 5
#le=loudness_energy
le_pearson_r,le_pearson_p_val=stats.pearsonr(data['loudness'], data['energy'])
le_spearman_r,le_spearman_p_val=stats.spearmanr(data['loudness'], data['energy'])

print('Pearson Coefficient: ',le_pearson_r,"; P-value: ",le_pearson_p_val)
print('Spearman Coefficient: ',le_spearman_r,"; P-value: ",le_spearman_p_val)

plt.scatter(data['loudness'], data['energy'],color='magenta')
plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.title('Loudness vs. Energy')
plt.show()

#Question 6
r2_scores={}
mses={}
coefs={}
y_preds={}
for feature in features:
    X = data[[feature]]
    y = data['popularity']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_preds[feature] = y_pred
    r2 = r2_score(y, y_pred)
    r2_scores[feature] = r2
    mse = mean_squared_error(y, y_pred)
    mses[feature] = mse
    coef = model.coef_
    coefs[feature]=coef
best_predictor_feature = max(r2_scores, key=r2_scores.get)
best_r2_score = r2_scores[best_predictor_feature]
print("Best predictor feature:", best_predictor_feature)
print("R-squared:", best_r2_score)
print("Coefficient: ", coefs[best_predictor_feature])
print("Mean squared error: ", mses[best_predictor_feature])
print("RMSE: ",np.sqrt(mses[best_predictor_feature]))
plt.scatter(data[[best_predictor_feature]], y, color="lavender", label='data')
plt.plot(data[[best_predictor_feature]], y_preds[best_predictor_feature], color="purple", linewidth=3, label='best fit')
plt.xlabel(best_predictor_feature)
plt.ylabel('popularity')
plt.title('Simple Linear Regression')
plt.legend(loc='upper right')
plt.show()

#Question 7
X_features = features
y_pop = data['popularity']
model = LinearRegression()
model.fit(X_features, y_pop)
y_pred = model.predict(X_features)
r2 = r2_score(y_pop, y_pred)
n = len(y_pop)
k = X_features.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print("R-squared of multiple linear regression model:", r2)
print("Adjusted R-squared of MLR model:",adjusted_r2)
print("Coefficient:", model.coef_)
print("Mean squared error:", mean_squared_error(y_pop, y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_pop, y_pred)))

#Question 8 
zscoredData=stats.zscore(features)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
rotatedData = pca.fit_transform(zscoredData)
varExplained = eigVals/sum(eigVals)*100
for j in range(len(varExplained)):
    print(varExplained[j].round(3))
kaiserThreshold = 1
print("# of factors selected by Kaiser criterion:", np.count_nonzero(eigVals > kaiserThreshold))
print("Variance accounted for by principal components: ",np.sum(varExplained[0:3]))
x = np.linspace(1,10,10)
plt.bar(x, eigVals, color='pink')
plt.plot([0,10],[1,1],color='purple') 
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Elbow Criterion')
plt.show()

#Question 9

X_valence = data[['valence']]
y_mode = data['mode']  
model = LogisticRegression()
model.fit(X_valence, y_mode)
y_pred = model.predict(X_valence)
print("Classification Report:")
print(classification_report(y_mode, y_pred,zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_mode, y_pred))
plt.figure(figsize=(10, 6))
plt.scatter(X_valence, y_mode, color='blue', label='Actual')
plt.scatter(X_valence, y_pred, color='red', marker='x', label='Predicted')
plt.plot(X_valence, model.predict_proba(X_valence)[:,1], color='green', linewidth=3, label='Logistic Regression')
plt.title('Logistic Regression: Actual vs Predicted')
plt.xlabel('Valence')
plt.ylabel('Mode (Major:1, Minor:0)')
plt.legend()
plt.grid(True)
plt.show()
auroc = roc_auc_score(y_mode, model.predict_proba(X_valence)[:,1])
print("AUROC:", auroc)
fpr, tpr, thresholds = roc_curve(y_mode, model.predict_proba(X_valence)[:,1])
plt.plot(fpr, tpr, color='salmon', lw=2, label='ROC Curve (AUROC = %0.4f)' % auroc)
plt.plot([0, 1], [0, 1], color='purple', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve -- valence')
plt.legend(loc="lower right")
plt.show()

features_to_test = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo','popularity','explicit',]
best_feature = None
best_auroc = 0.0

for feature in features_to_test:
    X = data[[feature]]
    model = LogisticRegression()
    model.fit(X, y_mode)
    y_pred_proba = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y_mode, y_pred_proba)
    if auroc > best_auroc:
        best_auroc = auroc
        best_feature = feature

print("Best Feature:", best_feature)
print("Best AUROC Score:", best_auroc)

X_best = data[[best_feature]]
model_best = LogisticRegression()
model_best.fit(X_best, y_mode)
y_pred_proba_best = model_best.predict_proba(X_best)[:, 1]
fpr_best, tpr_best, thresholds_best = roc_curve(y_mode, y_pred_proba_best)
plt.plot(fpr_best, tpr_best, color='orange', lw=2, label='ROC curve (AUROC = %0.4f)' % best_auroc)
plt.plot([0, 1], [0, 1], color='magenta', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve -- speechiness')
plt.legend(loc="lower right")
plt.show()

#Question 10
pca_features = rotatedData[:,:3]

label_encoder = LabelEncoder()
data['is_classical'] = label_encoder.fit_transform(data['track_genre'] == 'classical')

X_duration = data[['duration']]
X_pca = pca_features 
y_classical = data['is_classical']

model_duration = LogisticRegression()
model_pca = LogisticRegression()

model_duration.fit(X_duration, y_classical)
model_pca.fit(X_pca, y_classical)

auroc_duration = roc_auc_score(y_classical, model_duration.predict_proba(X_duration)[:,1])
auroc_pca = roc_auc_score(y_classical, model_pca.predict_proba(X_pca)[:,1])

print("AUROC for Duration Predictor:", auroc_duration)
print("AUROC for PCA Predictor:", auroc_pca)

#EXTRA CREDIT
    
data['explicit'] = data['explicit'].astype(int)
X_energy = data[['energy']]
y_explicit = data['explicit']

genre_popularity = data.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)

plt.figure(figsize=(20, 12))
genre_popularity.plot(kind='bar', color='skyblue')
plt.title('Average Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

genre_groups = [data[data['track_genre'] == genre]['popularity'] for genre in data['track_genre'].unique()]
anova_result = stats.f_oneway(*genre_groups)

print("ANOVA results:")
print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)

model_energy = LogisticRegression()
model_energy.fit(X_energy, y_explicit)
y_pred_proba=model_energy.predict_proba(X_energy)[:,1]
auroc_energy = roc_auc_score(y_explicit,y_pred_proba)
fpr_energy, tpr_energy, thresholds_energy = roc_curve(y_explicit, y_pred_proba)
plt.plot(fpr_energy, tpr_energy, color='pink', lw=2, label='ROC curve (AUROC = %0.4f)' % auroc_energy)
plt.plot([0, 1], [0, 1], color='magenta', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve -- energy - explicitness')
plt.legend(loc="lower right")
plt.show()

X_loudness=data[['loudness']]
model_loudness = LogisticRegression()
model_loudness.fit(X_loudness, y_explicit)
y_pred_proba=model_loudness.predict_proba(X_loudness)[:,1]
auroc_loudness = roc_auc_score(y_explicit,y_pred_proba)
fpr_loudness, tpr_loudness, thresholds_loudness = roc_curve(y_explicit, y_pred_proba)
plt.plot(fpr_loudness, tpr_loudness, color='red', lw=2, label='ROC curve (AUROC = %0.4f)' % auroc_loudness)
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve -- loudness - explicitness')
plt.legend(loc="lower right")
plt.show()

