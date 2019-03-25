# Wasserstein Autoencoder
Tensorflow implementation of deterministic Wasserstein Autoencoder[1] and Adversarial Autoencoder[2] with MNIST[3]

## Warning

Stochastic Wasserstein AutoEncoder is invalid now

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
> [2] Makhzani, A., Shlens, J., Jaitly, N., & Goodfellow, I.J. (2015). Adversarial Autoencoders. *CoRR, abs/1511.05644*.
>
> [3] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*, 2278-2324.