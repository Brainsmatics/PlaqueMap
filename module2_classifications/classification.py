'''

Morphological Classification (Diffuse, Core, CAA)

Comprising two parts: model training and prediction.
An example of the operation can be found in the main function.

'''



import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import plot_importance
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from contextlib import redirect_stdout



features_pick = ["shape_VoxelVolume", "shape_MeshVolume", "shape_SurfaceArea", "shape_SurfaceVolumeRatio", "shape_Compactness1", "shape_Compactness2", "shape_Sphericity",
                 "shape_SphericalDisproportion", "shape_Maximum3DDiameter", "shape_Maximum2DDiameterSlice", "shape_Maximum2DDiameterColumn", "shape_Maximum2DDiameterRow",
                 "shape_MajorAxisLength", "shape_MinorAxisLength", "shape_LeastAxisLength", "shape_Elongation", "shape_Flatness", "glcm_Autocorrelation", "glcm_JointAverage",
                 "glcm_ClusterProminence", "glcm_ClusterShade", "glcm_ClusterTendency", "glcm_Contrast", "glcm_Correlation", "glcm_DifferenceAverage", "glcm_DifferenceEntropy",
                 "glcm_DifferenceVariance", "glcm_JointEnergy", "glcm_JointEntropy", "glcm_Imc1", "glcm_Imc2", "glcm_Idm", "glcm_Idmn", "glcm_Id", "glcm_Idn", "glcm_InverseVariance",
                 "glcm_MaximumProbability", "glcm_SumEntropy", "glcm_SumSquares", "glszm_GrayLevelVariance", "glszm_ZoneVariance", "glszm_SizeZoneNonUniformity",
                 "glszm_GrayLevelNonUniformity", "glszm_SmallAreaHighGrayLevelEmphasis", "glszm_LargeAreaLowGrayLevelEmphasis", "glszm_LargeAreaHighGrayLevelEmphasis",
                 "glszm_SmallAreaLowGrayLevelEmphasis", "glszm_SmallAreaEmphasis", "glszm_LargeAreaEmphasis", "glszm_GrayLevelNonUniformityNormalized",
                 "glszm_SizeZoneNonUniformityNormalized", "glszm_ZonePercentage", "glszm_ZoneEntropy", "glszm_LowGrayLevelZoneEmphasis", "glszm_HighGrayLevelZoneEmphasis",
                 "ngtdm_Coarseness", "ngtdm_Contrast", "ngtdm_Busyness", "ngtdm_Complexity", "ngtdm_Strength", "gldm_SmallDependenceEmphasis", "gldm_LargeDependenceEmphasis",
                 "gldm_GrayLevelNonUniformity", "gldm_DependenceNonUniformity", "gldm_DependenceNonUniformityNormalized", "gldm_GrayLevelVariance", "gldm_DependenceVariance",
                 "gldm_DependenceEntropy", "gldm_LowGrayLevelEmphasis", "gldm_HighGrayLevelEmphasis", "gldm_SmallDependenceLowGrayLevelEmphasis",
                 "gldm_SmallDependenceHighGrayLevelEmphasis", "gldm_LargeDependenceLowGrayLevelEmphasis", "gldm_LargeDependenceHighGrayLevelEmphasis"]


def train_data(f_dir):
    features_all = pd.read_csv(f_dir)
    index = features_all.columns
    # cls = pd.DataFrame()
    X = features_all[features_pick]
    Y = features_all["cls"]
    for i in range(Y.shape[0]):
        if Y.iloc[i] == 1.5:
            Y.iloc[i] = 1

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    x_mini, x_minitest, y_mini, y_minitest = train_test_split(x_test, y_test, test_size=0.1)

    return X, Y, x_train, x_test, y_train, y_test, x_mini, x_minitest, y_mini, y_minitest



def compare_without_training(x_mini, x_minitest, y_mini, y_minitest, output_dir):
    x = x_mini
    y = y_mini
    x_ = x_minitest
    y_ = y_minitest

    from sklearn.model_selection import cross_val_score
    models = {
        'XGBoost': XGBClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'GaussianNB': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'SVC': SVC()
    }

    scores = {}

    with open(output_dir + "compare_score_notrain.txt", 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            for name, model in models.items():
                cv_scores = cross_val_score(model, x, y, cv=2, scoring='accuracy')
                scores[name] = cv_scores.mean()
                print(f"{name:20s}: {cv_scores.mean():.4f} (Â±{cv_scores.std():.4f})")
    return scores



def compare(x_mini, x_minitest, y_mini, y_minitest, output_dir):
    x = x_mini
    y = y_mini
    x_ = x_minitest
    y_ = y_minitest

    # Other models compared on mini-batches
    xgbc_model=XGBClassifier()
    # RandomForest
    rfc_model=RandomForestClassifier()
    # ET
    et_model=ExtraTreesClassifier()
    # Naive Bayes
    gnb_model=GaussianNB()
    # KNN
    knn_model=KNeighborsClassifier()
    # Logistic regression
    lr_model=LogisticRegression()
    # Decision Tree
    dt_model=DecisionTreeClassifier()
    # SVM
    svc_model=SVC()

    xgbc_model.fit(x,y)
    rfc_model.fit(x,y.astype(int))
    et_model.fit(x,y.astype(int))
    gnb_model.fit(x,y.astype(int))
    knn_model.fit(x,y.astype(int))
    lr_model.fit(x,y.astype(int))
    dt_model.fit(x,y.astype(int))
    svc_model.fit(x,y.astype(int))

    # Accuracy obtained using 2-fold cross-validation (the mean accuracy across each iteration)
    xgb_score = cross_val_score(xgbc_model, x_, y_, cv=2).mean()
    rfc_score = cross_val_score(rfc_model, x_, y_, cv=2).mean()
    et_score = cross_val_score(et_model, x_, y_, cv=2).mean()
    gnb_score = cross_val_score(gnb_model, x_, y_, cv=2).mean()
    knn_score = cross_val_score(knn_model, x_, y_, cv=2).mean()
    lr_score = cross_val_score(lr_model, x_, y_, cv=2).mean()
    dt_score = cross_val_score(dt_model, x_, y_, cv=2).mean()
    svc_score = cross_val_score(svc_model, x_, y_, cv=2).mean()

    score =[['xgb_score: ', 'rfc_score: ', 'et_score: ', 'gnb_score: ',
             'knn_score: ', 'lr_score: ', 'dt_score: ', 'svc_score: '],
            [xgb_score, rfc_score, et_score, gnb_score,
             knn_score, lr_score, dt_score, svc_score]]
    score = pd.DataFrame(score)
    score.T.to_csv(output_dir+"compare_score.txt", index=False, header=False)



# Model training
def xgb_model(X, Y, x_train, x_test, y_train, y_test, output_dir):
    # During model training, parameters require continual debugging and comparison to obtain the optimal parameter combination.
    model = XGBClassifier(
        max_depth=30,   # The maximum depth of the tree.The greater the max_depth, the more specific and localised samples the model will learn.
        learning_rate=0.1,
        n_estimators=100, # Number of weak classifiers used, number of iterations.
        objective='multi:softmax',    # Multiple classifications.
        num_class=6,
        booster='gbtree',    # Trees, or gblinear.
        gamma=0,             # Gamma specifies the minimum loss function decrease required for node splitting.
        min_child_weight=5,      # Determine the minimum leaf node sample weight to prevent overfitting.
        max_delta_step=0,       # Limit the maximum step size for weight changes per tree, default 0.
        subsample=0.8,           # The proportion of the total sample used for each decision tree. Controls the proportion of random sampling for each tree.
        colsample_bytree=0.8,    # Typical values for the proportion of random sampling of features during tree construction (applied to features).
        reg_alpha=0,            # The L1 regularisation term for the weights.
        reg_lambda=1,           # The L2 regularisation term for the weights.
        seed=None                # Random number seed.
    )
    model.fit(x_train,y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
            eval_metric='mlogloss', verbose=50, early_stopping_rounds=100)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    y_pred_train = model.predict(x_train)
    accuracy1 = accuracy_score(y_train,y_pred_train)
    Y_pred_all = model.predict(X)
    accuracy2 = accuracy_score(Y, Y_pred_all)
    print("test_accuracy: ", accuracy, "////train_accuracy: ", accuracy1, "////all_accuracy: ", accuracy2)

    precision = precision_score(y_test,y_pred,average='micro')
    precision1 = precision_score(y_train, y_pred_train,average='micro')
    precision2 = precision_score(Y, Y_pred_all,average='micro')
    print("test_precision: ", precision, "////train_precision: ", precision1, "////all_precision: ", precision2)

    recall = recall_score(y_test, y_pred, average='micro')
    recall1 = recall_score(y_train, y_pred_train, average='micro')
    recall2 = recall_score(Y, Y_pred_all, average='micro')
    print("test_recall: ", recall, "////train_recall: ", recall1, "////all_recall: ", recall2)

    f1 = f1_score(y_test, y_pred, average='micro')
    f1_score1 = f1_score(y_train, y_pred_train, average='micro')
    f1_score2 = f1_score(Y, Y_pred_all,average='micro')
    print("test_f1: ", f1, "////train_f1: ", f1_score1, "////all_f1: ", f1_score2)

    # Display key features
    feature_importance = model.get_booster().get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_ipt = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    feature_ipt.to_csv(output_dir+'f_importance.csv', index=False, header=False)
    fig, ax = plt.subplots(figsize=(25, 25))
    plot_importance(model, height=0.5, ax=ax, max_num_features=75)
    # plot_importance(model)
    # plt.show()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(output_dir+"f_all.jpg")

    model.save_model(model_dir+'f_all.model')

    clf = model.get_booster()
    clf.dump_model(model_dir+'dump-f_all.txt')

    pickle.dump(model, open(model_dir+"f_all.pickle.dat", "wb"))



def updata_xgb_model(x_train, y_train, ori_dir):
    # The model is retrained using the new data.
    params = {
        'max_depth': 12,
        'learning_rate': 0.1,
        'n_estimators': 140,
        'objective': 'multi:softmax',
        'num_class': 3,
        'booster': 'gbtree',
        'gamma': 2,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 0.9,
        'colsample_bytree': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'seed': 10
    }
    y_train_ = y_train - 1
    booster = xgb.Booster(params)
    loaded_model = booster.load_model(ori_dir)     # Import the original model
    dtrain_new = xgb.DMatrix(x_train, y_train_)

    model_updated = xgb.train(
        params=params,
        dtrain=dtrain_new,   # dtrain_new
        num_boost_round=50,  # Number of newly added trees
        xgb_model=loaded_model,  # Continue training from the original model
        evals=[(dtrain_new, 'train')],
        verbose_eval=True
    )

    model_updated.save_model(model_dir + 'updata_model.model')

    clf = model_updated.get_booster()
    clf.dump_model(model_dir + 'updata_model_dump.txt')

    pickle.dump(model_updated, open(model_dir + "updata_model.pickle.dat", "wb"))




# Model parameter debugging
def param_test(x_mini, y_mini, output_dir):
    x = x_mini
    y = y_mini

    # Stepwise parameter tuning: begin with a larger learning rate parameter of lr=0.1, then fine-tune the learning rate.
    # 1) Adjust max_depth and min_child_weight
    # max_depth is typically set between 3 and 10.
    param_test12 = {
        'max_depth': [i for i in range(3, 11, 1)],
        'min_child_weight': [i for i in range(0, 7, 1)]
    }
    # 2) Gamma parameter tuning
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 51)]
    }
    # 3) Adjust the subsample and colsample_bytree parameters
    param_test45 = {
        'subsample':[i/10.0 for i in range(5,11)],
        'colsample_bytree':[i/10.0 for i in range(5,11)]
    }
    # 4) Regularisation parameter tuning reg_alpha
    param_test6 = {
        'reg_alpha':[0, 1e-5, 1e-2, 0.1, 1, 100]
    }
    # 5) Regularisation parameter tuning reg_lambda
    param_test7 = {
        'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100]
    }
    # Iteration count selection
    n_e = {'n_estimators': [i for i in range(10, 310, 10)]}
    # Seed count selection
    # n_seed = {'seed': [i for i in range(0, 101, 1)]}
    n_seed = {'seed': [None, 0 ,60]}      #seed_new
    # Learning rate selection
    lr = {'learning_rate': [0, 1e-5, 1e-2, 5e-2, 0.1, 0.5, 1]}

    para_list = ["param_test12", "param_test3", "param_test45", "param_test6", "param_test7", "n_e", "n_seed", "lr"]
    for para_pick in para_list:
        if para_pick == "param_test12":
            name = "max_depth-min_child_weight"
            para = param_test12
        if para_pick == "param_test3":
            name = "gamma"
            para = param_test3
        if para_pick == "param_test45":
            name = "subsample-colsample_bytree"
            para = param_test45
        if para_pick == "param_test6":
            name = "reg_alpha"
            para = param_test6
        if para_pick == "param_test7":
            name = "reg_lambda"
            para = param_test7
        if para_pick == "n_e":
            name = "n_estimators"
            para = n_e
        if para_pick == "n_seed":
            name = "seed"
            para = n_seed
        if para_pick == "lr":
            name = "learning_rate"
            para = lr

        param_out = pd.DataFrame()
        gsearch = GridSearchCV(
            estimator=XGBClassifier(
                max_depth=3,
                learning_rate=1e-4,
                n_estimators=100,
                objective='multi:softmax',
                num_class=3,
                booster='gbtree',
                gamma=0,
                min_child_weight=0,
                max_delta_step=0,
                subsample=0.5,
                colsample_bytree=0.5,
                reg_alpha=0,
                reg_lambda=1,
                seed=None),
            param_grid=para,    # Each parameter adjustment requires a separate modification.
            scoring="accuracy",
            n_jobs=4,
            cv=5)
        gsearch.fit(x.values, y.values)
        param_out.insert(loc=0, column='cv_results_', value=gsearch.cv_results_)
        param_out.insert(loc=1, column='cv_results_value', value=gsearch.cv_results_.values())
        param_out.to_csv(output_dir + name + ".txt", index=False)   # Each parameter adjustment requires a separate modification.
        param_out2 = [['best_params_: ', 'best_score_: '],
                      [gsearch.best_params_, gsearch.best_score_]]
        param_out2 = pd.DataFrame(param_out2)
        param_out2.T.to_csv(output_dir + name + "-2.txt", index=False, header=False)
        print('gsearch1.cv_results_', gsearch.cv_results_, "\n")
        print('gsearch1.best_params_', gsearch.best_params_, "\n")
        print('gsearch1.best_score_', gsearch.best_score_, "\n")


def convert_string_to_complex(series):
    def convert_value(x):
        if pd.isna(x):
            return np.nan
        elif isinstance(x, (int, float, complex)):
            return complex(x)
        elif isinstance(x, str):
            try:
                # Clean up the string
                s = x.strip().replace(' ', '').replace('i', 'j')
                return complex(s)
            except (ValueError, TypeError):
                return np.nan
        else:
            return np.nan
    return series.apply(convert_value)



def process_dataframe_complex_columns(df: object, columns: object) -> object:
    # Processing complex values in specified columns of a DataFrame.
    if columns is None:
        # Automatically detect columns that may contain multiple values.
        columns = df.select_dtypes(include=[object, complex]).columns
    result_df = df.copy()
    for col in columns:
        result_df[col] = result_df[col].apply(
            lambda x: x[1:-1] if isinstance(x, str) and len(x) > 2 else x
        )
        result_df[col] = convert_string_to_complex(result_df[col])
        result_df[col] = result_df[col].apply(
            lambda x: abs(x) if isinstance(x, complex) else float(x)
        )
    return result_df


def data_prepare(f_dir):
    # Predictive data preparation.
    # features_all = pd.read_csv(f_dir)
    features_all = pd.read_excel(f_dir)

    '''
    # If complex values are present in features_all, adjustments shall be made.
    
    processed_data = process_dataframe_complex_columns(features_all, ['shape_MajorAxisLength', 'shape_MinorAxisLength',
                                                                      'shape_LeastAxisLength', 'shape_Elongation', 'shape_Flatness'])
    '''
    data = features_all[features_pick]
    # data = processed_data[f_list]    # complex values exist
    ID = features_all["ID"]

    return features_all, data, ID
    # return processed_data, data, ID   # complex values exist



def model_load(model_, data, ID, out_dir):
    clf = xgb.XGBClassifier()
    clf.load_model(model_)

    # loaded_model = pickle.load(open(model_, "rb"))
    data = np.array(data)
    y_pred = clf.predict(data)
    y_pred_1 = [round(value) for value in clf.predict(data)]
    y_pred_proba = clf.predict_proba(data)
    cls = pd.DataFrame()
    cls["ID"] = ID
    cls["cls_non"] = y_pred
    cls.to_csv(out_dir+"cls_nonlinear.csv")
    cls_1 = pd.DataFrame()
    cls_1["ID"] = ID
    cls_1["cls_non_"] = y_pred_1
    cls_1.to_csv(out_dir + "cls_nonlinear-round.csv")
    # cls_proba = pd.DataFrame()
    # cls_proba["ID"] = ID
    # cls_proba["cls_proba"] = y_pred_proba
    # cls_proba.to_csv(out_dir + "cls_nonlinear-proba.csv")
    # print('y_predï¼š', y_pred)
    # print('y_pred_probaï¼š', y_pred_proba)
    return cls


def linear_cls(features_all, ID, out_dir):
    cls_out_3 = []
    cls_out_4 = []
    conf_out = []

    # Classification results are corrected using the output from a linear classifier (previously trained).
    for i in range(features_all.shape[0]):
        features = features_all.iloc[i, :]
        cls1 = 1-(8.93037E-16)*features["shape_SurfaceVolumeRatio"]-(5.20881E-15)*features["shape_Sphericity"]\
               -(2.81253E-16)*features["shape_SphericalDisproportion"]+(2.40591E-19)*features["glcm_ClusterProminence"]\
               -(2.54578E-18)*features["glcm_ClusterShade"]+(2.16048E-15)*features["glcm_Correlation"]\
               +(4.79354E-15)*features["glcm_Imc1"]+(2.15717E-16)*features["glszm_ZoneEntropy"]\
               +(1.32239E-17)*features["glszm_HighGrayLevelZoneEmphasis"]\
               +(1.09159E-18)*features["gldm_LargeDependenceEmphasis"]-(2.28177E-15)*features["gldm_DependenceEntropy"]
        cls2 = 2+(3.49699E-19)*features["shape_VoxelVolume"]+(2.04572E-20)*features["shape_MeshVolume"]\
               -(6.94358E-19)*features["shape_SurfaceArea"]-(3.07964E-14)*features["shape_SurfaceVolumeRatio"]\
               -(4.26287E-15)*features["shape_Sphericity"]+(7.46057E-15)*features["shape_SphericalDisproportion"]\
               +(2.56324E-16)*features["shape_Maximum3DDiameter"]-(2.10292E-16)*features["shape_Maximum2DDiameterSlice"]\
               -(1.70349E-16)*features["shape_Maximum2DDiameterColumn"]+(1.40927E-16)*features["shape_Maximum2DDiameterRow"]\
               +(2.02676E-16)*features["shape_MajorAxisLength"]-(4.61044E-16)*features["shape_MinorAxisLength"]\
               -(3.01605E-16)*features["shape_LeastAxisLength"]+(8.16922E-15)*features["shape_Elongation"]\
               -(1.44764E-18)*features["glcm_ClusterProminence"]+(2.87612E-17)*features["glcm_ClusterShade"]\
               +(1.92355E-16)*features["glcm_ClusterTendency"]-(9.38449E-16)*features["glcm_Correlation"]\
               +(5.83485E-14)*features["glcm_Imc1"]-(6.26468E-22)*features["glszm_LargeAreaHighGrayLevelEmphasis"]\
               +(1.01861E-15)*features["glszm_ZoneEntropy"]-(1.68754E-18)*features["glszm_HighGrayLevelZoneEmphasis"]\
               +(2.26041E-11)*features["ngtdm_Coarseness"]+(7.97374E-18)*features["ngtdm_Busyness"]\
               -(5.12316E-14)*features["ngtdm_Strength"]+(2.10176E-18)*features["gldm_DependenceNonUniformity"]\
               +(1.15329E-16)*features["gldm_DependenceVariance"].astype(float)-(1.43223E-14)*features["gldm_DependenceEntropy"].astype(float)
        cls3 = 3+(4.19594E-15)*features["shape_SurfaceVolumeRatio"]-(2.27221E-14)*features["shape_Sphericity"]\
               -(5.48691E-16)*features["shape_SphericalDisproportion"]-(1.99957E-15)*features["shape_Elongation"]\
               -(9.81719E-16)*features["glcm_Autocorrelation"]+(2.65808E-14)*features["glcm_JointAverage"]\
               +(1.64948E-10)*features["glcm_ClusterTendency"]+(1.64985E-10)*features["glcm_Contrast"]\
               -(5.11044E-15)*features["glcm_Correlation"]-(2.46226E-15)*features["glcm_JointEntropy"]\
               -(1.09288E-14)*features["glcm_Imc1"]-(3.72454E-15)*features["glcm_MaximumProbability"]\
               -(6.57274E-16)*features["glcm_SumEntropy"]-(6.59797E-10)*features["glcm_SumSquares"]\
               -(2.02216E-13)*features["glcm_DifferenceAverage"]+(2.53153E-14)*features["glcm_DifferenceEntropy"]\
               -(2.14689E-14)*features["glcm_DifferenceVariance"]-(3.10833E-13)*features["glcm_Idm"]\
               -(8.42514E-13)*features["glcm_Idmn"]+(4.65832E-13)*features["glcm_Idn"]\
               +(1.99795E-16)*features["glszm_GrayLevelVariance"]+(1.73002E-15)*features["glszm_ZonePercentage"]\
               +(9.76368E-16)*features["glszm_ZoneEntropy"]-(1.02966E-16)*features["glszm_HighGrayLevelZoneEmphasis"]\
               -(3.85352E-13)*features["ngtdm_Contrast"]-(1.94945E-16)*features["ngtdm_Complexity"]\
               +(6.54006E-15)*features["gldm_GrayLevelVariance"]-(3.79955E-15)*features["gldm_DependenceEntropy"]\
               -(5.28176E-14)*features["gldm_LowGrayLevelEmphasis"]-(4.0975E-16)*features["gldm_HighGrayLevelEmphasis"]\
               -(5.281E-16)*features["gldm_SmallDependenceHighGrayLevelEmphasis"]\
               +(6.92152E-17)*features["gldm_LargeDependenceLowGrayLevelEmphasis"]\
               +(3.62863E-19)*features["gldm_LargeDependenceHighGrayLevelEmphasis"]

        cls4_1 = 1 - (0.0000000000000009002) * features["shape_SurfaceVolumeRatio"] - (6.83722E-15) * features["shape_Sphericity"] \
                 - (1.47919E-16) * features["shape_SphericalDisproportion"] - (1.20186E-19) * features["glcm_ClusterProminence"] \
                 - (6.98554E-19) * features["glcm_ClusterShade"] - (0.000000000052854) * features["glcm_ClusterTendency"] \
                 - (0.0000000000528537) * features["glcm_Contrast"] + (8.80347E-16) * features["glcm_Correlation"] + 5.03975E-16 * features["glcm_Imc1"] \
                 + (0.000000000211416) * features["glcm_SumSquares"] + (1.36047E-15) * features["glszm_SizeZoneNonUniformityNormalized"] \
                 - (7.65234E-17) * features["glszm_ZoneEntropy"] + (2.83481E-17) * features["glszm_HighGrayLevelZoneEmphasis"] \
                 + (1.96362E-17) * features["gldm_LargeDependenceEmphasis"] + (7.63109E-16) * features["gldm_DependenceEntropy"] \
                 + (9.64652E-15) * features["gldm_LowGrayLevelEmphasis"] - (3.14603E-17) * features["gldm_LargeDependenceLowGrayLevelEmphasis"]
        cls4_2 = 2 - (9.48361E-19) * features["shape_VoxelVolume"] + (3.75511E-19) * features["shape_MeshVolume"] + (6.62484E-20) * features["shape_SurfaceArea"] \
                 - (5.72378E-15) * features["shape_SurfaceVolumeRatio"] - (6.92513E-14) * features["shape_Sphericity"] + (1.26183E-15) * features["shape_SphericalDisproportion"] \
                 + (2.90902E-16) * features["shape_Maximum3DDiameter"] - (2.03756E-16) * features["shape_Maximum2DDiameterSlice"] \
                 - (1.47514E-16) * features["shape_Maximum2DDiameterColumn"] + (1.5577E-16) * features["shape_Maximum2DDiameterRow"] \
                 + (1.46913E-16) * features["shape_MajorAxisLength"] + (6.34657E-17) * features["shape_MinorAxisLength"] - (3.56692E-18) * features["shape_LeastAxisLength"] \
                 - (5.36286E-18) * features["glcm_ClusterProminence"] + (9.08643E-17) * features["glcm_ClusterShade"] - (2.21657E-11) * features["glcm_ClusterTendency"] \
                 - (2.2172E-11) * features["glcm_Contrast"] + (5.76791E-15) * features["glcm_Correlation"] - (2.16417E-14) * features["glcm_Imc1"] \
                 - (7.23328E-14) * features["glcm_Imc2"] - (1.16677E-12) * features["glcm_Idmn"] + (8.86617E-11) * features["glcm_SumSquares"] \
                 + (4.46841E-19) * features["glszm_ZoneVariance"] + (7.66263E-16) * features["glszm_GrayLevelVariance"] + (2.33461E-17) * features["glszm_SizeZoneNonUniformity"] \
                 - (4.27098E-17) * features["glszm_GrayLevelNonUniformity"] - (6.84568E-16) * features["glszm_SmallAreaHighGrayLevelEmphasis"] \
                 - (4.31259E-19) * features["glszm_LargeAreaLowGrayLevelEmphasis"] - (1.54001E-21) * features["glszm_LargeAreaHighGrayLevelEmphasis"] \
                 - (6.96204E-14) * features["glszm_SizeZoneNonUniformityNormalized"] - (8.62486E-15) * features["glszm_ZoneEntropy"] \
                 + (4.10915E-16) * features["glszm_HighGrayLevelZoneEmphasis"] + (2.71665E-11) * features["ngtdm_Coarseness"] + (5.55136E-17) * features["ngtdm_Busyness"] \
                 - (7.58128E-14) * features["ngtdm_Strength"] - (3.88603E-20) * features["gldm_GrayLevelNonUniformity"] - (4.95304E-18) * features["gldm_DependenceNonUniformity"] \
                 + (2.58461E-15) * features["gldm_GrayLevelVariance"] - (1.30701E-14) * features["gldm_DependenceEntropy"] + (9.48272E-19) * features["gldm_LargeDependenceHighGrayLevelEmphasis"]
        cls4_3 = 3 + (4.57971E-15) * features["shape_SurfaceVolumeRatio"] + (9.14026E-19) * features["shape_SphericalDisproportion"] \
                 - (1.53473E-15) * features["shape_Elongation"] - (7.24596E-15) * features["glcm_JointAverage"] \
                 - (6.66158E-18) * features["glcm_ClusterProminence"] + (5.80843E-17) * features["glcm_ClusterShade"] \
                 + (1.19641E-10) * features["glcm_ClusterTendency"] + (1.19672E-10) * features["glcm_Contrast"] \
                 - (1.52291E-15) * features["glcm_Correlation"] - (8.81923E-14) * features["glcm_DifferenceAverage"] \
                 + (6.74774E-16) * features["glcm_DifferenceEntropy"] - (2.0005E-14) * features["glcm_DifferenceVariance"] \
                 - (4.78103E-14) * features["glcm_JointEnergy"] + (4.75954E-15) * features["glcm_JointEntropy"] \
                 + (5.41067E-15) * features["glcm_Imc1"] - (6.62672E-15) * features["glcm_Imc2"] - (3.42412E-12) * features["glcm_Idm"] \
                 - (1.08567E-12) * features["glcm_Idmn"] + (4.18152E-12) * features["glcm_Id"] + (6.97205E-13) * features["glcm_Idn"] \
                 + (3.54029E-13) * features["glcm_InverseVariance"] + (7.79915E-15) * features["glcm_SumEntropy"] \
                 - (4.78563E-10) * features["glcm_SumSquares"] - (7.14872E-15) * features["glszm_GrayLevelNonUniformityNormalized"] \
                 - (1.6453E-15) * features["glszm_SizeZoneNonUniformityNormalized"] - (5.51442E-16) * features["glszm_ZoneEntropy"] \
                 - (8.60111E-18) * features["glszm_HighGrayLevelZoneEmphasis"] - (2.60418E-13) * features["ngtdm_Contrast"] \
                 - (1.78248E-16) * features["ngtdm_Complexity"] - (1.29057E-14) * features["gldm_SmallDependenceEmphasis"] \
                 + (3.23585E-16) * features["gldm_DependenceVariance"] - (9.74125E-15) * features["gldm_DependenceEntropy"] \
                 - (1.17328E-13) * features["gldm_LowGrayLevelEmphasis"] + (7.51901E-16) * features["gldm_HighGrayLevelEmphasis"] \
                 + (1.06604E-15) * features["gldm_SmallDependenceHighGrayLevelEmphasis"] + (4.44239E-17) * features["gldm_LargeDependenceLowGrayLevelEmphasis"]
        cls4_4 = 4 + (0.000000000000034616) * features["shape_SurfaceVolumeRatio"] + (0.00000000000075348) * features["shape_Compactness1"] \
                 + (0.0000000000000336312) * features["shape_Compactness2"] - (4.49094E-15) * features["shape_SphericalDisproportion"] \
                 + (0.0000000000000167352) * features["glcm_JointAverage"] + (3.37767E-10) * features["glcm_ClusterTendency"] \
                 + (0.000000000337771) * features["glcm_Contrast"] - (0.0000000000000388056) * features["glcm_JointEnergy"] - (0.0000000000000167412) * features["glcm_Imc1"] \
                 + (0.000000000000667851) * features["glcm_Idmn"] - (0.00000000135107) * features["glcm_SumSquares"] - (6.93759E-15) * features["glszm_SmallAreaEmphasis"] \
                 + (0.0000000000000731759) * features["glszm_SizeZoneNonUniformityNormalized"] + (5.55886E-15) * features["glszm_ZoneEntropy"] \
                 - (1.10519E-16) * features["glszm_HighGrayLevelZoneEmphasis"] + (0.0000000000000117443) * features["ngtdm_Contrast"] \
                 - (0.0000000000000145259) * features["ngtdm_Strength"] - (0.000000000000380963) * features["gldm_SmallDependenceEmphasis"] \
                 + (4.90941E-16) * features["gldm_DependenceVariance"] - (0.0000000000000157022) * features["gldm_DependenceEntropy"] - (6.44741E-16) * features["gldm_HighGrayLevelEmphasis"]

        min1 = abs(cls1 - 1)
        min2 = abs(cls2 - 2)
        min3 = abs(cls3 - 3)
        # min_ = min(min1, min2, min3)
        min_ = min(min1, min2, min3)
        if min_ == min1:
            cls = 1
            conf = 0.7
        elif min_ == min2:
            cls = 2
            conf = 0.7
        elif min_ == min3:
            cls = 3
            conf = 0.7
        else:
            cls = 0
            conf = 0
        cls_out_3.append(cls)
        conf_out.append(conf)

        min1 = abs(cls4_1 - 1)
        min2 = abs(cls4_2 - 2)
        min3 = abs(cls4_3 - 3)
        min4 = abs(cls4_4 - 4)
        # min_ = min(min1, min2, min3)
        min_ = min(min1, min2, min3, min4)
        if min_ == min1:
            cls = 1
        elif min_ == min2:
            cls = 2
        elif min_ == min3:
            cls = 3
        elif min_ == min4:
            cls = 4
        cls_out_4.append(cls)

    cls_out = pd.DataFrame()
    cls_out["ID"] = ID
    cls_out["cls_3"] = cls_out_3
    cls_out["cls_4"] = cls_out_4
    cls_out["conf"] = conf_out
    cls_out.to_csv(out_dir + "cls_linear.csv")
    return cls_out


def cls_correct(features_all, ID, cls_non, cls_l, out_dir):
    # correct_cls
    conf_new = []
    cls_new = []
    for i in range(cls_non.shape[0]):
        cls_non_p = cls_non.iloc[i, :]
        cls_l_p = cls_l.iloc[i, :]
        features_p = features_all.iloc[i, :]
        bool_p = features_p[features_pick].eq(0).all()
        if cls_non_p["ID"] == cls_l_p["ID"] == features_p["ID"]:
            cls_nonlinear = cls_non_p["cls_non"] + 1     # cls_nan = 0,1,2
            cls_l1 = cls_l_p["cls_3"]
            cls_l2 = cls_l_p["cls_4"]
            cls_conf = cls_l_p["conf"]

            # Classification results are corrected using low-level features.
            ratio = features_p["ratio"]
            max_gray = features_p["max_gray"]

            cls_out = cls_nonlinear
            conf_out = cls_conf

            # gray peak test
            if max_gray >= 225:
                if cls_out == 2:
                    conf_out = conf_out + 0.1
                    if conf_out >= 1:
                        conf_out = 1
                else:
                    conf_out = conf_out - 0.1
            # axial ratio test
            if ratio >= 2.4:
                if cls_nonlinear != 3:
                    cls_out = 3
                else:
                    if ratio >= 2.6:
                        conf_out = cls_conf + 0.1

            # extreme circumstances
            if cls_l1 == 2 or cls_l2 == 2:
                cls_out = 2
                conf_out = conf_out - 0.1
            if cls_nonlinear == cls_l1 == cls_l2:
                conf_out = conf_out + 0.1
            elif (cls_nonlinear == cls_l1) or (cls_nonlinear == cls_l2):
                conf_out = conf_out
            else:
                if (cls_l2 == 4) & (cls_nonlinear == 1):
                    conf_out = conf_out
                else:
                    conf_out = conf_out - 0.1

            if bool_p == True:
                conf_out = 0

            if conf_out >= 1:
                conf_out = 1

            cls_new.append(cls_out)
            conf_new.append(conf_out)

    df_cls = pd.DataFrame()
    df_cls["ID"] = ID
    df_cls["cls"] = cls_new
    df_cls["conf"] = conf_new
    # df_cls.insert(loc=df_cls.shape[1], column='ID', value=ID)
    # df_cls.sort_values("ID", inplace=True)
    df_cls.to_csv(out_dir + 'cls_correct.csv')
    return df_cls



if __name__ == '__main__':
    model_dir = "./cls_model/"
    base_dir = "./cls_test/"

    ''' Model testing and training '''
    # train_out = "./cls_model/"
    # X, Y, x_train, x_test, y_train, y_test, x_mini, x_minitest, y_mini, y_minitest = train_data(f_n_dir)
    # compare_without_training(x_mini, x_minitest, y_mini, y_minitest, train_out)
    # compare(x_mini, x_minitest, y_mini, y_minitest, train_out)
    # param_test(x_mini, y_mini, train_out)
    # xgb_model(X, Y, x_train, x_test, y_train, y_test, train_out)
    # # updata_xgb_model(x_train, y_train, ori_dir)    # The model can be retrained.

    ''' Model prediction '''
    model_ = model_dir + "final2_model.model"

    pre_f_dir = base_dir + "test.xlsx"    # Check for the presence of complex values.
    features_all, data, ID = data_prepare(pre_f_dir)
    cls_non = model_load(model_, data, ID, base_dir)   # cls_non = 0(diffuse), 1(core), 2(caa)
    cls_l = linear_cls(features_all, ID, base_dir)   # Ensure all required features are included in features_all. cls_l = 1(diffuse), 2(core), 3(caa)

    # output: df_cls.  cls = 1(diffuse), 2(core), 3(caa)
    df_cls = cls_correct(features_all, ID, cls_non, cls_l, base_dir) # When the conf value is less than 0.5, manual correction may be considered.






