# playing-cifar10-pytorch
playing-cifar10-pytorch


### Requirement
<pre>
<code>
 # easydict
 pip install easydict

 # pytorch-gradual-warmup-lr
 pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
 
 # cutmix
 pip install git+https://github.com/ildoonet/cutmix
</code>
</pre>
    
    

### Repo Structure
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
<pre>
<code>
 python main.py --params experiment_1_cutmix.json
</code>
</pre>
