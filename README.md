In this project, I completed varying types of analysis on the provided dataset to determine aspects like correlation between variables and the strength of different variables as predictors. Fortunately, there are no NaN values in this dataset so I did not see any benefit in cleaning the data. However, I did implement dimension reduction via PCA for the relevant portions of the project. Libraries and frameworks I utilized were Pandas, NumPy, SciPy, Matplotlib, and Scikit-learn.
1. Consider the 10 song features duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. Is any of these features reasonably distributed normally? If so, which one?

I first created a subplot grid with 2 rows and 5 columns, and an overall figure size of 30 by 12, intended to store a histogram for each of the selected features. The fig variable holds the Figure object, while the axes variable holds an array of Axes objects corresponding to each subplot. I then flattened the Axes array, reshaping it from two dimensions to one, making it easier to iterate over. I did so using a for loop and the enumerate() function allowing me to access the index and feature name simultaneously. I chose bins=51 as it is an adequate value to convey the distribution without overcrowding the plot. After creating and labeling all subplots, I used plt.tight_layout() to prevent any overlap.
Observing Figure 1, the only feature that holds a relatively normal distribution is danceability (seen in the 2nd column of the first row of Figure 1).
2. Is there a relationship between song length and popularity of a song? If so, is the relationship positive or negative?

To gain insight into the type of relationship, if any, between duration and popularity, I plotted the values as a scatter plot (Figure 2). In my opinion, this plot indicates a potential negative relationship. However, for further confirmation, I calculated both the Pearson and Spearman correlation coefficients. My program returned a value of approximately -0.055 for 𝑟 and -0.037 for ρ, thus indicating a weak, negative relationship between duration and popularity.

3. Are explicitly rated songs more popular than songs that are not explicit?

 
 I created two groups of popularity scores, differentiated by whether the songs where explicit or not. I then calculated the mean popularity for both groups and conducted a Mann-Whitney U test to determine whether the difference in popularity between explicit and non-explicit songs is significant or simply due to chance. I opted for this particular significance test rather than a t-test because the assumptions of the t-test were not met. This test returned a U-statistic of 139361273.5 and a p-value of 3.0679199339114678e-19. The U-statistic indicates the difference in the sum of ranks of explicit songs and non-explicit songs. Assuming an alpha value of 0.05, the p-value of this test is well below that, suggesting that the difference between groups is statistically significant. Now, considering the means of the groups, where explicit songs have a mean popularity of approximately 35.81 and non-explicit a mean of 32.79, I have concluded that explicitly rated songs are more popular than songs that are not explicit.
4. Are songs in major key more popular than songs in minor key?
Here, I took a nearly identical approach to the previous question. Instead, the two groups of popularity scores were differentiated by their mode, a binary variable in which 1 indicates the major key, and 0 indicates the minor key. Once again, I calculated the mean of each group and completed a Mann-Whitney U test. This returned a U-statistic of 309702373.0 and a p-value of 2.0175287554899416e-06. Once again, assuming an alpha value of 0.05, the results of the test suggest that the group means are, in fact, different, and not by chance. The mean popularity of songs in the major key is approximately 32.76 and for songs in the minor key it is 33.71, thus I have concluded that songs in the minor key are more popular.
5. Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute) that this is the case?
Kondamani 3

 Kondamani 4
 Figure 3: Scatter plot of loudness versus energy
To explore the relationship between loudness and energy, I first plotted the two on a scatterplot (Figure 3). There is a positive monotone relationship between the two variables. To further substantiate this belief, I calculated both the Pearson and Spearman correlation coefficients, as it is hard to determine whether this relationship is linear or not. My program returned a value of approximately 0.775 for 𝑟 and 0.731 for ρ, thus indicating a strong positive relationship between loudness and energy. Thus it is fair to assert that energy reflects the loudness of a song.
6. Which of the 10 song features in question 1 predicts popularity best? How good is this model?
 
 Figure 4: Linear regression model predicting popularity from instrumentalness
To determine which of the 10 features best predicts popularity I wrote a short program to iterate through each feature and create a linear regression model. I started by initializing four dictionaries (r2_scores, mses, coefs, and y_preds) to store results related to each feature's performance. Then I created a for loop to begin the actual iteration, in which each feature acts as the X and the y, popularity, remains the same. Each time, a model is created, fitted, and the target
variable is predicted from the feature. For each feature, the coefficient of determination (𝑅2), mean squared error (MSE), coefficient of the model, and predicted values of y are calculated and stored in their respective dictionaries. After iterating over all features, I determined the feature
that produces the highest 𝑅2value using max(r2_scores, key=r2_scores.get) along with the associated coefficient, MSE, and RMSE.
According to the program, the best predictor was instrumentalness with an approximate
𝑅2 of 0.021, coefficient of -9.691, MSE of 462.843, and RMSE of 21.514. I then created a scatter plot of instrumentalness against popularity and plotted the best-fit line obtained from the regression model on top (Figure 4). As seen in the figure and from the associated values, although instrumentalness was the best of the predictors, it is not necessarily a good predictor.
The 𝑅2 value indicates that instrumentalness only explains around 2.1% of the variance in popularity, which is rather low. Additionally, given that popularity can range from 0 to 100, the RMSE is relatively high, suggesting the predictions made by the model are not very accurate. In conclusion, instrumentalness is the best predictor of our 10 features but is overall not a good predictor for popularity.
Kondamani 5

 7. Building a model that uses *all* of the song features in question 1, how well can you predict popularity? How much (if at all) is this model improved compared to the model in question 7). How do you account for this?
Here, I utilized multiple linear regression, simply creating a model where y remains the
same (popularity) and X is set to all 10 song features. Like before, I created and fitted the model and then calculated the 𝑅2, MSE, RMSE, and coefficients. My program returned approximate
values with an 𝑅2 of 0.0477, MSE of 450.237, RMSE of 21.219, and coefficients [-8.12966659e-06, 5.05399676e+00, -1.38413219e+01, 6.62993824e-01, -7.38828256e+00, 9.78514738e-01, -8.77191669e+00, -2.34193433e+00, -8.10241388e+00, 8.40718143e-03]. I
also calculated the adjusted 𝑅2, using the formula 1 − (1−𝑅2)(𝑁−1) , to ensure that this new 𝑁−𝑘−1
model is not falsely optimistic. I received a value of 0.0475, which is very close to the 𝑅2 value.
From the previous model, 𝑅2 has more than doubled, while RMSE has remained very similar, only slightly decreasing. This change can be accounted for by the introduction of multiple features, thus different sources of variability in popularity. Now, the model is more
versatile. Objectively, 𝑅2 is still low which makes perfect sense given that in the previous model,
despite its low 𝑅2, instrumentalness was still the best predictor of the 10 features.
8. When considering the 10 song features above, how many meaningful principal
components can you extract? What proportion of the variance do these principal components account for?
Kondamani 6
 
 Kondamani 7
 Figure 5: Selection of meaningful principal components via Elbow Criterion
To determine how many meaningful principal components exist among the ten features, I first z-scored the data, normalizing each feature. I then applied Principal Component Analysis to the standardized data and retrieved the eigenvalues of the principal components, I also computed the variance accounted for by each component. After this, I transformed the standardized data into the new coordinate system defined by the principal components, effectively reducing the dimensionality. Finally, I used the Kaiser criterion to determine the number of meaningful factors and received a value of 3, which I also plotted as a bar graph (Figure 5). Thus, I could extract 3 meaningful components.
These 3 components explain approximately 57.36% of the variance, a value that I computed by summing the first 3 values of the array that contained variance accounted for by each component.
9. Can you predict whether a song is in major or minor key from valence? If so, how good is this prediction? If not, is there a better predictor?

 Figure 6: Logistic regression, predicting mode from valence
Figure 7: ROC Curve for logistic regression model for valence
 
 Kondamani 9
 Figure 8: ROC Curve for logistic regression model for speechiness
To predict whether a song’s key is major or minor from valence, I used the mode feature, which assigns a binary value to distinguish the key of a song. I then created and fitted a logistic regression model, the results of which can be seen in Figure 6. To further determine the quality of this model, with valence as a predictor, I calculated the area under the receiver operating characteristics curve (AUROC). I received a value of approximately 0.5072, which is extremely low, implying that this model is close to being a random classifier, and therefore, not a good predictor (Figure 7).
In search of a better predictor, I iterated through the rest of the features, such as energy and loudness, and for each feature, I created a logistic regression model to predict the mode. I calculated the AUROC of each model and returned the highest value along with its corresponding feature. I found that the best predictor was speechiness, with an AUROC value of approximately 0.5631 (Figure 8). Although this is still not an ideal model, it is significantly better than using valence as a predictor.
10. Which is a better predictor of whether a song is classical music – duration or the principal components you extracted in question 8?

 To answer this question, I first used Scikit-learn’s label encoder to transform the genre label to a binary value representing whether or not a song is classical. I then created two logistic regression models, one in which X was set to the feature duration, and another in which it was the 3 principal components previously extracted. I then fit both models and calculated AUROC for both models. Using duration as a predictor returned a value of approximately 0.581 whereas the principal components as a predictor returned approximately 0.945. Given these results, it is clear that the 3 principal components acted as a far better predictor for whether or not a song is classical than duration.
11. Extra credit: Tell us something interesting about this dataset that is not trivial and not already part of an answer (implied or explicitly) to these enumerated questions.
Figures 9, 10: ROC Curve for logistic regression predicting explicitness from energy and loudness respectively
From my experience, lots of music that can be considered louder and to have more
energy, what people call “hype” songs, tend to be explicit. Thus, I was curious to see whether these two features could act as good predictors of whether a song is explicit or not. To do this, I first transformed explicit values from booleans to binary values. I then created two separate models, one with energy as a predictor, and the other using loudness. I then calculate the
Kondamani 10
  
 AUROC of each model and plotted the curve (Figures 9 & 10). I found that loudness is a better predictor than energy, however neither obtains an ideal AUROC value.
Kondamani 11