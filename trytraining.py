#!/usr/bin/env python
# coding: utf-8

# In[173]:


import pandas as pd
import gc
import re
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
import datasets
import transformers
print(transformers.__version__)
from pathlib import Path
import torchaudio

import warnings 
warnings.filterwarnings("ignore")

from tqdm import tqdm
tqdm.pandas()


# In[174]:


RATE_HZ = 16000 # resampling rate in Hz
MAX_LENGTH = 160000 # maximum audio interval length to consider (= RATE_HZ * SECONDS)
AUDIO_FOLDER = Path("fma_test") 


# In[175]:


import json
lableid_file = 'fma_metadata\mappings.json'

# Read mappings back from the JSON file
with open(lableid_file, 'r') as f:
    mappings = json.load(f)
    label2id = mappings['label2id']
    id2label = mappings['id2label']

# Print loaded mappings to confirm
print("Loaded label2id:", label2id)
print("Loaded id2label:", id2label)
print(len(label2id))


# In[176]:


from pathlib import Path
import torchaudio
with open('fma_metadata\\track_genres.json', 'r') as f:
    data = json.load(f)
df = [{"file": track_id, "label": genre_id} for track_id, genre_id in data.items()]
dd = pd.DataFrame(df)
print(dd.head())
print(dd.shape)
print(dd['label'].value_counts())


# In[177]:


def get_transform_audio(file):
    audio,rate = torchaudio.load(str(file))
    transform = torchaudio.transforms.Resample(rate,RATE_HZ)
    if audio.size(0) > 1:  # 如果是多通道
        audio = audio.mean(dim=0)
    audio = transform(audio).numpy()
    audio = audio[:MAX_LENGTH]
    return audio # truncate to first part of audio to save RAM


# In[178]:


audio_data = []
labels = []

for _, row in dd.iterrows():
    # 补齐文件名到6位
    track_id = row["file"].zfill(6)
    label = row["label"]
    
    # 拼接文件路径
    audio_path = AUDIO_FOLDER / track_id[:3] / f"{track_id}.mp3"
    
    if audio_path.exists():
        try:
            # 读取并处理音频
            audio = get_transform_audio(audio_path)
            audio_data.append(audio)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    else:
        print(f"Audio file not found: {audio_path}")

# 检查结果
print("Processed audio files:", len(audio_data))
print("Processed labels:", len(labels))


# In[179]:


print(type(labels))
CLASS_NUM=len(np.unique(labels))
print(CLASS_NUM)
labels = torch.tensor(labels, dtype=torch.long)
print("Tensor dtype:", labels.dtype)

totdd = pd.DataFrame({
    "label": labels,
    "audio": audio_data
})

# 检查结果
print(totdd.sample(5)) 
totdd["audio"] = totdd["audio"].apply(lambda x: np.array(x, dtype=np.float32))
print(totdd.sample(5)) 
print(totdd['label'].dtype)


# In[180]:


lengths = totdd["audio"].apply(len)
print(lengths.describe())

# 检查是否存在极端长的音频
print(lengths.nlargest(5))


# In[181]:


print(totdd["audio"].apply(len).nunique()) 
totdd = totdd[totdd["audio"].apply(len) >= 16000]
print(totdd["audio"].apply(len).nunique()) 


# In[182]:


from datasets import Dataset, ClassLabel
totdd = Dataset.from_pandas(totdd)

from collections import Counter
Counter(totdd['label']).items()


# In[183]:


print(type(totdd['label'][0]))
totdd = totdd.map(lambda x: {"label": torch.tensor(x["label"], dtype=torch.long)}, batched=True)
print(type(totdd['label'][0]))


# In[184]:


totdd = totdd.train_test_split(test_size=0.2)


# In[185]:


print(totdd)
totdd['train'] = totdd['train'].remove_columns(['__index_level_0__'])
totdd['test'] = totdd['test'].remove_columns(['__index_level_0__'])
totdd


# In[186]:


from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_str = "MIT/ast-finetuned-audioset-10-10-0.4593" 
feature_extractor = AutoFeatureExtractor.from_pretrained(model_str)
model = AutoModelForAudioClassification.from_pretrained(model_str,num_labels=len(label2id),ignore_mismatched_sizes=True)
model.config.id2label = id2label
# number of trainable parameters
print(model.num_parameters(only_trainable=True)/1e6)
print(len(label2id))
print(len(np.unique(labels)))


# In[187]:


def preprocess_function(batch):    
    inputs = feature_extractor(
        batch['audio'], 
        sampling_rate=RATE_HZ, 
        max_length=MAX_LENGTH, 
        truncation=True
    )
    inputs['input_values'] = inputs['input_values'][0]
    
    # Convert label to torch.long
    if 'label' in batch:
        inputs['labels'] = torch.tensor(batch['label'], dtype=torch.long)
    
    return inputs

totdd['test'] = totdd['test'].map(
    preprocess_function, 
    remove_columns=totdd['test'].column_names,
    batched=False
)

totdd['train'] = totdd['train'].map(
    preprocess_function, 
    remove_columns=totdd['train'].column_names,
    batched=False
)
totdd['train'].set_format(type='torch', columns=['input_values', 'labels'])
totdd['test'].set_format(type='torch', columns=['input_values', 'labels'])


# In[188]:


print("Training data label type:", type(totdd['train'][0]['labels']))
print("Training data label dtype:", totdd['train'][0]['labels'].dtype)
print("Test data label type:", type(totdd['test'][0]['labels']))
print("Test data label dtype:", totdd['test'][0]['labels'].dtype)


# In[189]:


gc.collect()


# In[190]:


import evaluate
from sklearn.preprocessing import label_binarize

accuracy = evaluate.load("accuracy")

from sklearn.metrics import roc_auc_score
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions  # shape: (n_samples, 163)
    label_ids = eval_pred.label_ids     # shape: (n_samples,)
    
    # 获取当前数据集中实际出现的类别
    present_classes = np.unique(label_ids)
    print(f"Present classes in current dataset: {present_classes}")
    print(f"Number of present classes: {len(present_classes)}")
    print(f"Shape of predictions: {predictions.shape}")
    
    # 应用softmax得到概率
    predictions = np.exp(predictions)/np.exp(predictions).sum(axis=1, keepdims=True)
    
    # 计算准确率（这个不需要修改，因为argmax会自动找到最大概率的类别）
    acc_score = accuracy.compute(
        predictions=predictions.argmax(axis=1),
        references=label_ids
    )['accuracy']
    
    # 对于ROC AUC，我们只考虑实际出现的类别
    # 将标签转换为二值化形式，但只针对出现的类别
    y_true_bin = label_binarize(label_ids, classes=present_classes)
    
    # 只取出对应出现类别的预测概率
    predictions_subset = predictions[:, present_classes]
    
    # 计算ROC AUC
    try:
        roc_auc = roc_auc_score(
            y_true=y_true_bin,
            y_score=predictions_subset,
            multi_class='ovr',
            average='macro'
        )
    except ValueError as e:
        print(f"ROC AUC calculation error: {e}")
        roc_auc = 0.0
    
    return {
        "roc_auc": roc_auc,
        "accuracy": acc_score,
        "present_classes": len(present_classes)
    }


# In[193]:


from transformers import TrainingArguments, Trainer
batch_size=4
warmup_steps=50
weight_decay=0.02
num_train_epochs=10
model_name = "test_train_model"
training_args = TrainingArguments(
    output_dir=model_name,
    logging_dir='./logs',
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=5e-5, # 5e-6
    logging_strategy='steps',
    logging_first_step=True,
    load_best_model_at_end=True,
    logging_steps=10,#1
    evaluation_strategy='epoch',
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    eval_steps=20,#1
    gradient_accumulation_steps=1, 
    gradient_checkpointing=True,
    save_strategy='epoch',
    save_total_limit=1, # save fewer checkpoints to limit used space
    report_to="mlflow",  # log to mlflow

    optim="adamw_hf",  # Optimizer, in this case Adam
    adam_beta1=0.9,  # Adam optimizer beta1
    adam_beta2=0.999,  # Adam optimizer beta2
    adam_epsilon=1e-8,  # Adam optimizer epsilon
    fp16=True,
    lr_scheduler_type="linear"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=totdd["train"],
    eval_dataset=totdd["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)


# In[195]:


trainer.evaluate()


# In[ ]:


trainer.train()

