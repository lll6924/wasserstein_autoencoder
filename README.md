# Wasserstein Autoencoder
Tensorflow implementation of deterministic Wasserstein Autoencoder[1] with MNIST[2]

## Prerequisites

```
pip install git+https://github.com/haowen-xu/tfsnippet.git
```

## Run the Code

```
python model.py [--args]
```

Run `python model.py -h` to check valid arguments

If you find the experiment intolerably slow, please reduce the conv_filters which is 128 by default

> [1] Tolstikhin, I.O., Bousquet, O., Gelly, S., & Schölkopf, B. (2017). Wasserstein Auto-Encoders. *CoRR, abs/1711.01558*.
>
> [2] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*, 2278-2324.