from openai import OpenAI
import os
import pandas as pd
import csv


client = OpenAI()


df = pd.read_csv("africmed_mcq.csv")
#data_preview = df["question"]
cq=df[['question','answer_options','correct_answer']]
questions=cq['question'].values.tolist()
correct_ans= cq['correct_answer'].values.tolist()
answers = []

for i in range(len(cq)):
    prompt=cq.iloc[i]['question']+". To answer the question, I am giving you options, only give option number in the format 'option number' as answer where number can be any option number. I am expecting answer from these options "+cq.iloc[i]['answer_options']
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a medical expert."},
                  {"role": "user", "content": prompt }]
    )
    answers.append(response.choices[0].message.content)

# Save to CSV
with open("answers.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Answer", "Correct Answer"])
    writer.writerows(zip(questions, answers, correct_ans))

print("File saved as answers.csv")