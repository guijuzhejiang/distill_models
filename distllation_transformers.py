from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification
import evaluate
import numpy as np
from transformers import DefaultDataCollator


teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

def process(examples):
    processed_inputs = teacher_processor(examples["image"])
    return processed_inputs

dataset = load_dataset("beans")

processed_datasets = dataset.map(process, batched=True)

class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device = Accelerator().device
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        student_output = self.student(**inputs)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        #temperature（温度）可以理解成“把老师的答案调得更软一点”的旋钮。它不是温度的物理概念，而是一个缩放系数：logits / T 再做 softmax。
        # T 越大，类别概率分布越平，top1 没那么“一边倒”；T 越小，分布越尖锐，更接近硬标签。这样做的好处是，学生能看到老师对各个类别的“相对偏好”，
        # 而不只是“最终猜哪个”。例如，假设老师对三类的 logits 是 [10, 2, 0]，当 T=1 时，概率几乎是“猫”独占；当 T=5 时，分布会变得明显更平，
        # 学生会看到“猫最高，但狗也不是完全没可能”。这就是蒸馏里常说的“soft targets”更有信息量。
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./distill_MobileNetV2",
    num_train_epochs=30,
    fp16=True,
    per_device_train_batch_size=160,
    per_device_eval_batch_size=160,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id="repo_name",
    # report_to="trackio",
    report_to="mlflow",
    run_name="distillation",
    )

num_labels = len(processed_datasets["train"].features["labels"].names)

# initialize models
teacher_model = AutoModelForImageClassification.from_pretrained(
    "merve/beans-vit-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# training MobileNetV2 from scratch
student_config = MobileNetV2Config()
student_config.num_labels = num_labels
student_model = MobileNetV2ForImageClassification(student_config)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}

data_collator = DefaultDataCollator()
trainer = ImageDistilTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    # training_args=training_args,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=data_collator,
    processing_class=teacher_processor,
    compute_metrics=compute_metrics,
    temperature=5,
    lambda_param=0.5
)

trainer.train()
trainer.evaluate(processed_datasets["test"])

save_dir = "./my-awesome-model/final"
trainer.save_model(save_dir)                 # 保存学生模型
teacher_processor.save_pretrained(save_dir)  # 保存图像处理器