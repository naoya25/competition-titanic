import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# データの読み込み
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
ex_df = pd.read_csv("data/gender_submission.csv")

# データの確認
# print(train_df.head())
# print(test_df.head())
# print(ex_df.head())

# 欠損値の確認
# print(train_df.isnull().sum())
# print(test_df.isnull().sum())


# 目的変数と説明変数の設定
# Xの情報からYの情報を予測する
# Xには一旦単純化のため、Pclass, SibSp, Parch, Sexの4つのみにしてるが、csvには他にもデータが入ってるよ
Y = train_df[["Survived"]]
X = train_df[["Pclass", "SibSp", "Parch", "Sex"]]

# ダミー変数化
# ワンホットエンコーディングすることで、データの偏りを減らすことができる
X = pd.get_dummies(X, columns=["Pclass", "Sex"])

# データの分割
# 本物のtestデータには、Survivedの情報がないので、精度を測ることができない
# そのため、trainデータを分割して、精度を測る
# 懸念点
# 1. 学習に使うデータ数がその分減るが、バーニーおじさんの法則により、問題ないと仮定しておく
# 2. データの分割の方法が適切かどうかは不明、データの性質によっては、データの偏りが大きくなる可能性がある
# X_train, Y_trainを使ってモデルを作成し、X_test, Y_testを使って精度を測る
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# モデルの作成
# ロジスティック回帰を使用
lr = LogisticRegression()
lr.fit(X_train, y_train)

# モデルの予測
y_pred = lr.predict(X_test)

# 精度の測定
print(lr.score(X_test, y_test))
# >> 0.7892376681614349
# 精度: 78.9%
# まあまあまあ、一旦OK

# 提出用データの作成
X_for_submit = test_df[["Pclass", "SibSp", "Parch", "Sex"]]
X_for_submit = pd.get_dummies(X_for_submit, columns=["Pclass", "Sex"])
pred_for_submit = lr.predict(X_for_submit)
submit = test_df[["PassengerId"]]
submit["Survived"] = pred_for_submit

submit.to_csv("./submission1/submission1.csv", index=False)
# 作成したsubmission1.csvを提出
# 自分で測った精度は78.9%となっていたが、実際の提出結果の精度とは違う
# 自分で測った精度はあくまで、train.csvから適当に分割したデータでの精度であり、test.csvの精度とは違うかた注意
# なので、提出結果と自分で測った精度が大きく違う場合は、データの分割の方法を見直す必要がある
