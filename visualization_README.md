## 测试和可视化

1. pip install -r requirements.txt (Python 2.7)
2. 按照README.md中方式，set up V-coco
3. 把输出的结果(results.txt.all)放在v-coco目录下
4. python test.py
5. 在集群环境里python vcoco_visualization.py {pickle_path} {res_dir}
   * pickle_path 是第4步中生成的fp_fn_samples.pkl的存放路径
   * res_dir是可视化图片的存储地址
   * 之所以要在集群上跑是因为我把coco图片路径hard code在代码里了