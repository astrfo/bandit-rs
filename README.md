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
`simulator.py`の`self.policy`で検証したいアルゴリズムを，`self.policy_plot_name`でそれに対応するアルゴリズム名を入れてください．
`self.policy`が辞書型になっているのはRS, RS-OPT, SRS, SRS-OPTの際にℵを設定できるようにするためです(l.57 setting参照してもらえると)．
CSVファイルをlogフォルダに保存するので`self.policy_plot_name`をいちいち変更するの面倒だなと感じる人はl.51をコメントアウトしてもらえれば動くと思います(多分)．
一応，簡素ですがCSVをプロットするプログラムは[ここ](https://github.com/astrfo/csv-plot)にあるので使ってどうぞ．

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

### 3. 以下のコマンドを実行する
```
python main.py
```

## 注意
SRS系のアルゴリズムは誤差によって確率が負になってエラー吐いたりします．それを解決したDecimal型のプログラムはいつか上げます．
