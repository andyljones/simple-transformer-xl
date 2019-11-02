This is a simple PyTorch implementation of [the Transformer XL](https://arxiv.org/abs/1901.02860), with everything but the core functionality thrown out. This makes it a lot easier to follow, understand, and adapt. 

You will want to have [the paper](https://arxiv.org/abs/1901.02860) to hand as you read over the code.

As well as [the implementation](transformer/transformer.py), there is [a simple demo to show that it can learn things LSTMs struggle with](transformer/demo.py).

### Credit
The code follows from [Huggingface's adaptation](https://github.com/huggingface/transformers), which in turn comes from the [CMU/Google Brain version](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch). Those both have a lot more features!