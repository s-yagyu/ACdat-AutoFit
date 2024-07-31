# ACdat-AutoFit

Version 0　20240731にVersion1に移行しました。

理研計器ACシリーズで測定したdatファイルをアップロードすると、自動で閾値を推定するお試し版のWebアプリです。
（あくまでもお試しですので、利用者の責任で閾値の最終判断をしてください。）

[https://s-yagyu-acdat-autofit-main-dat-autofit-wvzmv2.streamlit.app/](https://acdat-autofit-5ea7h7llssmahgky9kh5dv.streamlit.app/)

測定結果の1/2乗、1/3乗を計算して、拡張されたReLU関数をFitting関数として、最小化関数として絶対誤差を用いて推定を行っています。

それぞれのべき乗に対してFittingの当てはまり（決定係数）を評価し、決定係数が1に近い方を推奨としています。

R2が全体の決定係数、R2Sが閾値よりエネルギーが高い領域のみで計算した決定係数です。バックグラウンド成分に大きなノイズが乗っているときはR2Sの方の指標を優先した方がよいかもしれません。データの前処理機能（カウントがオーバーロードした際の飛び、ピークを含む場合など）は入っていませんので、その値に引きずられてまともな値が出ないことがあります。


なお、datファイルフォーマットの仕様および変換については理研計器社にお問い合わせください。

理研計器社が配布している変換ツールで

- AC-2、AC-3：（MDB型）変換ソフトで.datフォーマット（新形式0） へ変換
- AC-5：.datフォーマット（旧形式） 変換ソフトで.datフォーマット（新形式）へ変換

  
---

![ex00](./figs/ex00.JPG)

Datファイルをドラッグアンドドロップしてください。

----

#### Auのデータの例
![ex01](./figs/ex01.JPG)



Download Imageボタンを押すとグラフのダウンロードができます。

PYSの強度を1/2、1/3乗したグラフぞれぞれにReLU関数でのFittingを行います。そして決定係数を求めて比較します。
比較の結果、このデータの場合1/2乗の方が当てはまりがよいので、1/2乗で解析した閾値が選ばれています。なお、1/2乗のUserの線は人が引いた線です。

---

#### Si N型低抵抗のデータの例

![ex01](./figs/ex02.JPG)

Siの場合は、1/3乗の方が決定係数が1に近いのでこちらの値が選ばれています。
