# 満足化アルゴリズムにおけるバンディット問題シミュレーション
RS系のアルゴリズムと有名なのも少し入ってます．

## 使い方

### 1. パラメータを設定する
`main.py`にてシミュレーション回数(trial)，ステップ数(step)，腕の本数(K)を設定してください．
```
def main():
    trial = 1000
    step = 1000
    K = 2
```

### 2. 検証したいアルゴリズムを選択する
`simulator.py`の`self.policy`で検証したいアルゴリズムを入れてください．
`self.policy`が辞書型になっているのはRS, RS-OPT, SRS, SRS-OPTの際にℵを設定できるようにするためです(l.57 setting参照してもらえると)．

#### アルゴリズム一覧
- RS(ℵ=p_max)
- RS-OPT
- RS-CH
- SRS(ℵ=p_max)
- SRS-OPT
- SRS-CH
- ThompsonSampling
- UCB1
- UCB1-Tuned

### 3. コマンドを実行する
```
python main.py
```
結果のCSVファイルとパラメータを記載したテキストファイルをlogフォルダに保存する．
```
.
├── log
│   └── datetime(例：202209200909)
│       ├── xx.csv(例：RS.csv)
│       ├── yy.csv
│       └── zz.csv
```

### 4. 生成したCSVファイルをプロットする
グラフのタイトルとプロットしたいCSVが格納されているフォルダ名(例：202209200909)を指定してグラフのプロットを行います． 
生成したグラフはCSVと同じ階層にcsv_plot.pngという名前で保存されます．
```
python plot.py [グラフのtitle] [プロットしたいCSVが格納されているフォルダ名]
```

## 注意
**(Decimalなし)SRS系のアルゴリズムでたまに確率がNanになります**

ちょお待ちを．
