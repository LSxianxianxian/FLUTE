import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.pl_data_modules import DataModule
from src.pl_model_modules import MetaphorModel
from src.pl_callbacks import ClassificationMetricsCallback
import conf
import torch
import pandas as pd
from sklearn.metrics import f1_score
import os
import sys
torch.cuda.empty_cache()  # 清理缓存

print(">>> 当前 Python 执行文件：", os.path.abspath(__file__))
print(">>> 当前工作目录：", os.getcwd())
print(">>> Python 模块路径：", sys.path)

import src.wsd.dataset
print(">>> 加载的 dataset.py 路径：", src.wsd.dataset.__file__)

def main():
    # Seed for reproducibility
    pl.seed_everything(conf.seed)

    # Initialize TensorBoard Logger
    logger = TensorBoardLogger("logs/", name="metaphor_classification")

    # Initialize data module
    datamodule = DataModule(
        train_path=conf.train_path,
        val_path=conf.val_path,
        test_path=conf.test_path,
        batch_size=conf.batch_size,
    )

    # Initialize model
    model = MetaphorModel(
        model_name=conf.pmodel["name"],
        num_classes=4,
        lr=conf.lr,
    )

    # Callbacks
    callbacks = [
        ClassificationMetricsCallback(),
        ModelCheckpoint(
            monitor="val_loss",  # 监控验证集损失
            mode="min",  # 选择最小的 val_loss 作为最优模型
            save_top_k=1,  # 只保存最佳模型
            dirpath="checkpoints",
            filename="best_model",
            save_weights_only=True,  # **仅保存模型权重，防止加载问题**
        ),
    ]

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=conf.max_epochs,
        gpus=conf.gpus,
        precision=conf.precision,
        callbacks=callbacks,
        deterministic=conf.deterministic,
        logger=logger,  # **记录日志**
        log_every_n_steps=10,  # **每 10 步记录一次日志**
        accumulate_grad_batches=2,  # 每 2 个小 batch 才进行一次更新
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    test_results = []
    for dataset_name, test_path in conf.test_datasets.items():
        print(f"\n🔍 Testing on dataset: {dataset_name}")

        test_datamodule = DataModule(
            train_path=conf.train_path,
            val_path=conf.val_path,
            test_path=test_path,
            batch_size=conf.batch_size,
        )
        test_datamodule.setup(stage="test")
        test_datamodule.set_dataset_name(model)

        # 运行测试
        results = trainer.test(model, datamodule=test_datamodule)[0]


        # 记录 F1-score
        #results["F1-score"] = f1
        #results["dataset"] = dataset_name
        #test_results.append(results)


        # 记录测试结果
        results["dataset"] = dataset_name
        test_results.append(results)

        # 记录到 TensorBoard
        for metric, value in results.items():
            if metric != "dataset":
                logger.experiment.add_scalar(f"test/{dataset_name}/{metric}", value, global_step=trainer.global_step)


if __name__ == "__main__":
    main()