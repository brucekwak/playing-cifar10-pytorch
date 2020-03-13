# playing-cifar10-pytorch
playing-cifar10-pytorch


### Requirement
---------------------
[1] easydict
    $ pip install easydict
    
[2] pytorch-gradual-warmup-lr
    $ pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
    (https://github.com/ildoonet/pytorch-gradual-warmup-lr)

[3] cutmix
    $ pip install git+https://github.com/ildoonet/cutmix
    (https://github.com/ildoonet/cutmix)

### Repo Structure
---------------------
<pre>
<code>
|- data_loader.py   # dataset, data loader
|- main.py          # main 실행 파일
|- model.py         # model
|- train.py         # train, validate, test
|- utils.py
</code>
</pre>


### Run 
---------------------
$ python main.py --params experiment_1_cutmix.json