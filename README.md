# robotIntelliReport
For the Robot Intelligence report assignment.

## 実行環境
- Ubuntu18.04(機械系学科PC) / macOS Monterey12.0.1
- python2系

## 実行方法
### 1. リポジトリのクローン
```shell
git clone https://github.com/shin0805/robotIntelliReport.git
cd robotIntelliReport
```

### 2. データセットのダウンロード
```shell
python download_mnist.py
```
`data/`にpklファイルが出力される<br>
`picture/`に画像の例が出力される

### 3. 実行
```shell
python main.py
```
オプションは以下の通り
```txt
usage: main.py [-h] [--name NAME] [--noise NOISE]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -nm NAME
                        the name of trial
  --noise NOISE, -ns NOISE
                        the probability of noise [%]
```
例えば、訓練データに25%のノイズを乗せて、exampleという実験をする際は以下のように実行
```shell
python main.py -ns 25 -nm example
```
`result/example/`に結果のグラフが出力される<br>
`result/result.scv`に結果が書き込まれる<br>

ノイズを0%~25%まで付与して実験を行った後に以下のように実行するとノイズによる正答率の変化が`example/`に出力される
```shell
python util.py example
```



