# 安装环境
```bash
conda create -n itk python=3.11
conda activate itk
pip install -r requirements.txt
```

# 运行
```bash
python main.py --batch_size 12 --epochs 50 --lr 1e-4 --num_workers 8
```

# 可以用cache目录来缓存数据
```bash
python preprocess.py 

python main.py --use_cache --batch_size 4 --epochs 50 --lr 1e-4 --num_workers 8 
```
