# Data provenance

Download raw data locally using the eventual shared loader. Do not include raw datasets in the ZIP.
Commit checksums, training/validation index lists or reproducible generation recipes under manifests/.

Sources:
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Tiny Shakespeare: https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare

Pin an actual text commit/checksum when downloading. Split text before making windows.
