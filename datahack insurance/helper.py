import pandas as pd
from imblearn.over_sampling import SMOTE


def multiplyvals(x):
    ans = 1
    for i in x.split('@'):
        ans *= int("".join([j for j in i if j.isdigit()]))
    return ans


def preprocessing(df):
    for i in df.columns:
        if df[i][0] in ['Yes', 'No']:
            df[i] = df[i].map({"No": 0, "Yes": 1})
    special = ['max_power', 'max_torque']
    for i in special:
        df[i] = df[i].apply(multiplyvals)
    cols = ['area_cluster', 'segment', 'model', 'fuel_type',
            'engine_type', 'rear_brakes_type', 'transmission_type', 'steering_type']
    df = pd.get_dummies(df, columns=cols, drop_first=True)
    return df


def splitxy(df):
    X = df.drop(['policy_id', 'is_claim'], axis=1)
    y = df['is_claim']
    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(X, y)
    return X_sm, y_sm


def plot_fti(model, X):
    """
    Plot the feature importances of a model random forest
    model: a fitted random forest model
    X: dataframe without the target variable
    """
    ft = pd.DataFrame(X.columns, model.feature_importances_).sort_index(
        ascending=False).reset_index().set_index(0)
    import plotly.express as px
    return px.bar(ft)


def main():
    pass


if __name__ == '__main__':
    main()
