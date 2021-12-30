# Class project for Statistical Learning 2021
This is a personal repo of the Class project for **Statistical Learning 2021**. Here I provide all data as well as different methods codes.

I believe that basic methods are not supposed to cost lots of time, so just take and use these codes! Enjoy yourself, and Happy New Year!


## requirments
```
numpy
torch
PIL
matplotlib
scikit-learn
pandas
tqdm
```

## KNN & SVM
```
python run.py
--mode MODE  mode
--C C        svm's C
```

## GB & RF
All the code is in run.ipynb, just go ahead and get what you want.

## Conv
Note that I realize different cnn model in cnn.py, so maybe you need change the model by modify the line 210 to switch mode.
```
python cnn.py
```
This code will automaticlly save the best checkpoint in five fold cross validation and run the test procedure.

## Final Score
With the Tcnn1d, finally, the highest score is 0.96153. Note that all these models used for inference may not work with the best parameters. All these models may perform better with deliberated hyper-parameters.
![The score in Kaggle](/images/knn.png)
![](/images/svm.png)
![](/images/gbm.png)
![](/images/xgb.png)
![](/images/rf.png)
![](/images/cov1.png)
![](/images/cov2.png)