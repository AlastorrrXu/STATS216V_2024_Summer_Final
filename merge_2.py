import pandas as pd

# 加载性别信息数据
gender_info_file = r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-test.csv'
gender_df = pd.read_csv(gender_info_file)

# 性别列的名称
gender_column = 'Q1' 

# 加载预测数据
male_predictions_file = r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-predictions-nn-bs128-ep100-lr0.0001-dr0.7.csv'
female_predictions_file = r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-predictions-rf-n100-md10-ms2-ml1-mflog2.csv'

male_predictions_df = pd.read_csv(male_predictions_file)
female_predictions_df = pd.read_csv(female_predictions_file)

# 合并性别信息
male_merged_df = pd.merge(male_predictions_df, gender_df[['id_num', gender_column]], on='id_num', how='left')
female_merged_df = pd.merge(female_predictions_df, gender_df[['id_num', gender_column]], on='id_num', how='left')

# 分别提取男性和女性数据
male_results = male_merged_df[male_merged_df[gender_column].str.lower() == 'male']
female_results = female_merged_df[female_merged_df[gender_column].str.lower() == 'female']

# 合并男性和女性数据
final_results = pd.concat([male_results, female_results], ignore_index=True)

# 按照 id_num 排序
final_results = final_results.sort_values(by='id_num')

# 删除性别列
final_results = final_results.drop(columns=[gender_column])

# 保存最终的合并结果
output_file = r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-predictions-combined-gender.csv'
final_results.to_csv(output_file, index=False)

print(f"Saved combined results to {output_file}")
