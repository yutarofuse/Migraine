#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Import###
import pandas as pd
import numpy as np
import csv
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import glob
import os
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# In[3]:


##Questionnaire###
def get_keypoints_in_range(kp, des, height, width):
    filtered_kp = kp
    filtered_des = des
    return filtered_kp, np.array(filtered_des)

def align_images(img1, img2_path):
    img2 = cv2.imread(img2_path)

    detector = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    height, width, _ = img1.shape

    kp1, des1 = get_keypoints_in_range(kp1, des1, height, width)
    kp2, des2 = get_keypoints_in_range(kp2, des2, height, width)

    # Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    max_distance = 60

    good_matches = [m for m in matches if m.distance < max_distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    aligned_img2 = cv2.warpPerspective(img2, M, (width, height))

    return aligned_img2

# Base image
base_img_path = r"C:\Users\***\*sample.jpg"
img1 = cv2.imread(base_img_path)

directory = r"C:\Users\***"
file_patterns = ['M-*.png','M-*.tif', 'M-*.jpg']
for pattern in file_patterns:
    full_pattern = os.path.join(directory, pattern)
    for img2_path in glob.glob(full_pattern):
        if img2_path != base_img_path:
            aligned_img2 = align_images(img1, img2_path)
            cv2.imwrite(img2_path, aligned_img2)


# In[6]:


# Base Box location
file_path = r"C:\Users\***\*sample.txt"
with open(file_path, 'r') as f:
    lines = f.readlines()

location = []
result = []
for line in lines:
    values = line.strip().split(' ')
    values = [float(v) for v in values]
    location.append(values[1:])

location = np.array(location)

# Dataset creation

data = np.genfromtxt(r"C:\Users\***\M_names.csv", delimiter=',', dtype=None, encoding='utf-8')

df = pd.DataFrame(data)  
df = df.rename(columns={0:'Q',1:'A'})


# In[5]:


# Detection by YOLO-v7
# Inclusion of "--save-txt --save-conf" for the next step


# In[13]:


# Results of YOLO-v7
dir_path = r"C:\Users\***\labels"

for file_name in os.listdir(dir_path):
    if not file_name.endswith('.txt'):
        continue
    file_path = os.path.join(dir_path, file_name)
    
    with open(file_path, 'r') as f:
        data = f.readlines()

    label_1_boxes = []
    label_0_boxes = []

    for line in data:
        box = line.strip().split()
        if box[0] == '1' and (float(box[5]) >= 0.3):
            label_1_boxes.append(box)
        elif box[0] == '0':
            label_0_boxes.append(box)

    new_label_1_boxes = []
    for box_1 in label_1_boxes:
        overlap = False
        for box_0 in label_0_boxes:
            if abs(float(box_1[1]) - float(box_0[1])) < float(box_0[3])/2 and                abs(float(box_1[2]) - float(box_0[2])) < float(box_0[4])/2:
                overlap = True
                label_0_boxes.remove(box_0)
                break
        if not overlap:
            new_label_1_boxes.append(box_1)

    with open(file_path, 'w') as f:
        for box in label_0_boxes:
            f.write(' '.join(box) + '\n')


# In[7]:


files = glob.glob(r"C:\Users\***\labels\*.txt")
indivisualnames = [os.path.splitext(os.path.basename(p))[0] for p in files]

results = []
results2 = []
df = pd.DataFrame(df)

for i in indivisualnames:
    file_path =  r'C:\Users\***\labels\\' + str(i) + '.txt'
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        coords = []
        for line in lines:
            values = line.strip().split(' ')
            values = [float(v) for v in values]
            coords.append(values[1:])
        coords = np.array(coords)
        for x in location:
            x_center = coords[:, 0]
            y_center = coords[:, 1]
            prob = coords[:, 4]
            in_x = (
                ((x[0] - 0.5 * x[2]) <= x_center)
                & (x_center <= (x[0] + 0.5 * x[2]))
                & ((x[1] - 0.5 * x[3]) <= y_center)
                & (y_center <= (x[1] + 0.5 * x[3]))
                & (prob >= 0.5)
            )   
            results.append(int(any(in_x)))
            results2.append(prob[in_x].item() if any(in_x) else 0)

        df2 = pd.DataFrame({'indivisual': i, str(i): results}) 
        df2 = df2.iloc[:, 1]
        df = pd.concat([df, df2], axis=1)
        results = []
    except:
        df2 = pd.DataFrame({'indivisual': i, str(i): np.zeros(len(location), dtype=int)}) 
        df2 = df2.iloc[:, 1]
        df = pd.concat([df, df2], axis=1)
        results = []


# In[1]:


df.to_csv('M-AIprediction.csv', index=False, mode='w')
df


# In[16]:


# Directory for Answer Dataset
output_dir = r'C:/Users/***/answer/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
for col in df.columns[2:]:
    index = df[df[col] == 1].index
    result = location[index]
    new_result = np.insert(result, 0, 0, axis=1)
    filename = os.path.join(output_dir, col + '.txt')
    np.savetxt(filename, new_result, fmt=('%d', '%0.6f', '%0.6f', '%0.6f', '%0.6f'), delimiter=' ')


# In[20]:


# Answer Dataset

files = glob.glob(r"C:\Users\***\answer\*.txt")
indivisualnames = [os.path.splitext(os.path.basename(p))[0] for p in files]

results = []
df = pd.DataFrame(df)

for i in indivisualnames:
    file_path =  r'C:\Users\***\answer\\' + str(i) + '.txt'
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        coords = []
        for line in lines:
            values = line.strip().split(' ')
            values = [float(v) for v in values]
            coords.append(values[1:])
        coords = np.array(coords)
        for x in location:
            x_center = coords[:, 0]
            y_center = coords[:, 1]
            in_x = (
                ((x[0] - 0.5 * x[2]) <= x_center)
                & (x_center <= (x[0] + 0.5 * x[2]))
                & ((x[1] - 0.5 * x[3]) <= y_center)
                & (y_center <= (x[1] + 0.5 * x[3]))
            )   
            results.append(int(any(in_x)))
        df2 = pd.DataFrame({'indivisual': i, str(i): results}) 
        df2 = df2.iloc[:, 1]
        df = pd.concat([df, df2], axis=1)
        results = []
    except:
        df2 = pd.DataFrame({'indivisual': i, str(i): np.zeros(len(location), dtype=int)}) 
        df2 = df2.iloc[:, 1]
        df = pd.concat([df, df2], axis=1)
        results = []
df.to_csv('M-Answer.csv', index=False, mode='w')


# In[9]:


# Comparison of AIprediction and Answer Datasets
df = pd.read_csv("M-Answer.csv")
df2 = pd.read_csv("M-AIprediction.csv")

y_true_all_N = []
y_pred_all_N = []

for col in df2.columns[2:]:
    if col in df.columns:
        y_true_N = df[col].tolist()
        y_pred_N = df2[col].tolist()
        
        y_true_all_N.extend(y_true_N)
        y_pred_all_N.extend(y_pred_N)

acc_N = accuracy_score(y_true_all_N, y_pred_all_N)
acc_N
cm_N = confusion_matrix(y_true_all_N, y_pred_all_N)
auc_N = roc_auc_score(y_true_all_N, y_pred_all_N)

true_positive_N = cm_N[1, 1]
false_positive_N = cm_N[0, 1]
true_negative_N = cm_N[0, 0]
false_negative_N = cm_N[1, 0]

sensitivity_N = true_positive_N / (true_positive_N + false_negative_N)
specificity_N = true_negative_N / (true_negative_N + false_positive_N)
ppv_N = true_positive_N / (true_positive_N + false_positive_N)
fpr_N = false_positive_N / (false_positive_N + true_negative_N)
fnr_N = false_negative_N / (true_positive_N + false_negative_N)

# Results
print(f"Accuracy: {acc_N}")
print(f"Confusion Matrix:\n{cm_N}")
print(f"AUC-ROC: {auc_N}\n")
print(f"Sensitivity: {sensitivity_N}")
print(f"Specificity: {specificity_N}")
print(f"Positive Predictive Value (PPV): {ppv_N}")
print(f"False Positive Rate (FPR): {fpr_N}")
print(f"False Negative Rate (FNR): {fnr_N}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_N, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
plt.savefig("confusion_matrix_M_internal_new.png")


# In[23]:


# Create a dataset for ML

df2 = pd.read_csv("M-Answer.csv")

df2 = df2.T
df2.columns = df2.iloc[0] + df2.iloc[1]
df2 = df2.drop(df2.index[[0, 1]])
df2.index = df2.index.str.replace('M-', '')
df2 = df2.reset_index().rename(columns={'index': 'patient'})
df2.to_csv('M-Dataset.csv', index=True)


# In[ ]:




