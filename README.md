# TermProject_Econometrics
This is the term project of Advanced Econometrics II. I use three statistical learning models to predict the misstatement of listed companies’ financial statements. Before making any predictions, this paper first uses the SMOTE algorithm to oversample the original sample to deal with the unbalance problem in the original sample. I predict and evaluate logit models, CART, and Random Forest model. The research results show that the problem of imbalanced data does exist in the forecasting problem of misstatement of financial statements, and the SMOTE algorithm can indeed improve the precision of the three models; the accuracy of the traditional logit model in predicting the Chinese capital market is 67.3%, and the model accuracy of the decision tree and random forest is more than 80% with a larger AUC. Then I use the grid sereach method to tune the random forest model. The AUC of random forest model after tuning is 0.763.
