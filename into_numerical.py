import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取文件
train_file_path = r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-train.csv'
test_file_path = r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 将年龄字段转换为数值变量
def age_to_numeric(age):
    if pd.isna(age):
        return None
    elif age == 'Under 18':
        return 1
    elif age == '18-24':
        return 2
    elif age == '25-34':
        return 3
    elif age == '35-44':
        return 4
    elif age == '45-54':
        return 5
    elif age == '55-64':
        return 6
    elif age == '65 or older':
        return 7
    else:
        return None

# 检查是否存在年龄字段并进行转换
if '年龄' in train_data.columns:
    train_data['年龄'] = train_data['年龄'].apply(age_to_numeric)
if '年龄' in test_data.columns:
    test_data['年龄'] = test_data['年龄'].apply(age_to_numeric)

# 将分类变量转化为数值变量
label_encoders = {}
for column in train_data.columns[1:]:  # 跳过 'id_num' 列
    if train_data[column].dtype == 'object':
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

for column in test_data.columns[1:]:  # 跳过 'id_num' 列
    if test_data[column].dtype == 'object':
        le = label_encoders.get(column)
        if le:
            test_data[column] = le.transform(test_data[column])

# 保存处理后的数据
train_data.to_csv(r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-train-encoded.csv', index=False)
test_data.to_csv(r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-test-encoded.csv', index=False)
