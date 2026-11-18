import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

if __name__ == '__main__':
    # 定义要训练的模型列表
    model_names = ['LPNDNet']  # 根据您的实际需求修改
    
    for model_name in model_names:
        # 加载模型 - 假设您的模型文件命名规则是 4yuan1.yaml, 4yuan2.yaml 等
        model = YOLO(model=f'/root/{model_name}.yaml')
        
        # 注意：这里去掉了预训练权重的加载，保持与原代码一致
        # model.load('yolo11n.pt')  # 如果您需要加载预训练权重，可以在这里取消注释
        
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"{'='*60}")
        
        model.train(data='/root/precdataset/data.yaml',
                    imgsz=640,
                    epochs=300,
                    lr0=1e-3 * 0.3,  # 保持与原始代码相同的学习率
                    batch=4,
                    workers=4,
                    plots=True,
                    device='',
                    resume=False,
                    project='runs/train',  # 保持原始项目路径
                    name=model_name,  # 使用模型名称作为运行名称
                    single_cls=True,
                    amp=False
                    )
        
        print(f"\n{'='*60}")
        print(f"完成训练模型: {model_name}")
        print(f"{'='*60}")

    print("\n🎉 所有模型训练完成！")
    print(f"结果保存在: runs/train/ 目录下")

#focaler_iouD    Wise-MPDIoU