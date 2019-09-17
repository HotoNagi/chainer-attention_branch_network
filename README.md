# Chainer-Attention_Branch_Network

## train

```
python n_fold_train.py --dataset [DATASET] --batchsize [BATCHSIZE] --arch [MODEL]
```


論文：https://arxiv.org/abs/1812.10025
## どんなもの？
通常学習後に観測するActivation Mapを、Attentionとしてネットワーク内に組み込んだ研究。
![](https://i.imgur.com/evZdnaL.png)
Activation Mapの計算には特徴マップ以外にクラス分類への貢献を測る重みが必要だが(通常は全結合層の重みを使う)、これを取得するためAttention側からもクラス分類確率を出力し、マルチタスクで学習している。

## 先行研究と比べてどこがすごい？
Activation MapはGlabal Average Poolingで学習・推論する必要があり、特定の問題設定において性能が著しく低下する問題がある。
### Class Activation Mapping
![](https://i.imgur.com/8crjUWR.png)

CAM は Grad-CAM と異なり、勾配を利用するのではなく、CNN 層の後の Fully-Connected 層と一つの GAP（Global Average Pooling）に入れ替えています。この GAP は（豊富な特徴情報を持っている）最後の CNN の出力の特徴図（Feature Map）を Pooling して、分類のクラスとマッピングします（Class Activation Mapping）。

![](https://i.imgur.com/aSnlKM9.png)

### Global Average Pooling
![](https://i.imgur.com/hNMVFI9.png)
![](https://i.imgur.com/DYYDP1x.png)
Global Average Poolingをまとめた記事
https://qiita.com/mine820/items/1e49bca6d215ce88594a
http://pynote.hatenablog.com/entry/dl-pooling

本研究では可視化マップをAttention機構へ応用することで物体認識の性能を向上した。
* 1つのAttention mapのみに用いてネットワークを構築
* 認識における判断認識の視覚化に関する能力は残したままネットワークを構築

## 技術や手法のキモはどこ？
提案手法はAttention Branch Network（ABN）と呼ばれるもの。
### ネットワークの構築方法
* ベースのネットワークをFeature extractorとPerception branchに分割
* Feature extractorの後の層にAttention branchを導入

![](https://i.imgur.com/AVO226W.png)

Feature extractorとPerception branchの２つの学習誤差からネットワークを学習

![](https://i.imgur.com/21C8HGJ.png)

### Attention Branch Network（ABN）全体像
![](https://i.imgur.com/rn5zc4Z.png)

### Attention branchの構造
![](https://i.imgur.com/9kGVEDO.png)

畳み込み層とGlobal Average Poolingからのブランチを構築  

* Class Activation MappingとFully Convolitional Networkをベースにブランチを構築

Global Activation MappingとFully Convolutional Networkをベースにブランチを構築

### Perception branchの構造
Feature extractorの特徴マップに対してAttention mapを適応

![](https://i.imgur.com/nKRGxw3.png)

### Multi-task learningへの応用
そもそもマルチタスク学習とは？
<iframe src="//www.slideshare.net/slideshow/embed_code/key/m7AEcUN1OVQTUi?startSlide=3" width="510" height="420" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/DeepLearningJP2016/dlbeyond-shared-hierarchies-deep-multitask-learning-through-soft-layer-ordering" title="[DL輪読会]Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering " target="_blank">[DL輪読会]Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering </a> </strong> from <strong><a href="//www.slideshare.net/DeepLearningJP2016" target="_blank">Deep Learning JP</a></strong> </div>

タスク毎のPerception BranchにてAttention mapを用いて顔属性を認識
![](https://i.imgur.com/f8ngW27.png)

## どうやって有効だと検証した？
### ImageNetによる性能の比較
![](https://i.imgur.com/4mQX7IG.png)
### CIFAR100による性能の比較
![](https://i.imgur.com/I2Kebie.png)
### Comprehensive Cars Datasetによる評価
詳細認識における評価

![](https://i.imgur.com/yTHjdBO.png)
![](https://i.imgur.com/6zWq8mw.png)

Attention mapの可視化例

![](https://i.imgur.com/I59K3Nu.png)

残差機構の導入による効果

![](https://i.imgur.com/JUl0Rjl.png)

## 議論はある？
学習仮定にラベルを含まない強化学習にABNを適用する予定らしい
![](https://i.imgur.com/STfnPsI.png)

## 次に読むべき論文は？
[Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)

## 参考
* https://arxiv.org/abs/1812.10025
* https://www.slideshare.net/greentea1125/miru2018-global-average-poolingattention-branch-network
* https://www.slideshare.net/DeepLearningJP2016/dlbeyond-shared-hierarchies-deep-multitask-learning-through-soft-layer-ordering
* https://github.com/arXivTimes/arXivTimes/issues/1290
* https://github.com/machine-perception-robotics-group/attention_branch_network
