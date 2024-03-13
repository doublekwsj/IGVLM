import pandas as pd

# Step 1: Load the CSV data

csv_path = 'Please modify here'

df = pd.read_csv(csv_path)

def map_prediction_to_answer(row):
    # Extracting the letter part before the ":" delimiter if present
    answer_column = None
    if isinstance(row['pred'], str):
        prediction_letter = row['pred'][0]
        if prediction_letter in ["A","B","C","D","E"]:
            answer_column = 'a' + str(ord(prediction_letter) - ord('A'))
        if "answer is " in row['pred']:
            row["pred"] = row["pred"][row["pred"].index("answer is"):]
    if answer_column in ["a0","a1","a2","a3","a4"]:
        return row[answer_column]
    elif answer_column:
        print(prediction_letter)
    return "None"

#def set_answer_from_idx(row):
#    answer_idx = int(row['answer_idx'])
#    answer_column = f'a{answer_idx}'
#    return row[answer_column]
#df['answer'] = df.apply(set_answer_from_idx, axis=1)

df['predicted_answer'] = df.apply(map_prediction_to_answer, axis=1)

# Step 3: Measure Answer accuracy
df['is_correct'] = df['predicted_answer'] == df['answer']

total_accuracy = df['is_correct'].mean()

# Print total accuracy
print(f'Total Accuracy: {total_accuracy:.4f}')

# Step 4: Report group by question_type
accuracy_report = df.groupby('question_type')['is_correct'].mean()
print(accuracy_report)

#df['prefix'] = df['question_type'].apply(lambda x: x.split('_')[0])
df['prefix'] = df['question_type'].apply(lambda x: x[0])
grouped_accuracy = df.groupby('prefix')['is_correct'].mean()
print(grouped_accuracy)



