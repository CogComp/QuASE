import json
import numpy as np
from scipy.stats import bernoulli


def change_Large_QA_SRL_SQuAD(org_file, new_file):
    sent_num = 0
    qa_num_total = 0
    max_length = 0
    max_ques_length = 0
    fin = open(org_file)
    lines = fin.readlines()
    paragraphs = []
    for line in lines:
        point = json.loads(line)
        sentence = " ".join(point['sentenceTokens'])
        if len(point['sentenceTokens']) > max_length:
            max_length = len(point['sentenceTokens'])
        sentence_id = point['sentenceId']
        qas = []
        for key, value in point['verbEntries'].items():
            cur_question_index = -1
            for question, v in value['questionLabels'].items():
                cur_question_index += 1
                # valid_index = 0
                cur_answers = []
                for index in range(len(v['answerJudgments'])):
                    if v['answerJudgments'][index]['isValid']:
                        valid_index = index
                        # break
                        span_left = v['answerJudgments'][valid_index]['spans'][0][0]
                        span_right = v['answerJudgments'][valid_index]['spans'][0][1]
                        answer = " ".join(point['sentenceTokens'][int(span_left): int(span_right)])
                        answer_start = 0
                        for x in range(len(point['sentenceTokens']))[:int(span_left)]:
                            answer_start += len(point['sentenceTokens'][x])
                            answer_start += 1
                        cur_answers.append({'answer_start': answer_start, 'text': answer})
                qa_id = sentence_id + '-' + str(key) + '-' + str(cur_question_index) + '-' + str(valid_index)
                qa_num_total += 1
                if len(question.split()) > max_ques_length:
                    max_ques_length = len(question.split())
                qas.append({'answers': cur_answers, 'question': question, 'id': qa_id})
        sent_num += 1
        paragraphs.append({'context': sentence, 'qas': qas})
    data = {"data": [{'title': 'TQA', 'paragraphs': paragraphs}]}

    with open(new_file, 'w') as fin_new:
        json.dump(data, fin_new)
    print('max length',  max_length)
    print('max ques length', max_ques_length)
    print('qa total num', qa_num_total)
    print('sent num', sent_num)
    fin.close()


def sample_data(input_file, output_file, sample_rate):
    np.random.seed(seed=66)
    with open(input_file) as f:
        data = json.load(f)
    new_articles = []
    for article in data['data']:
        paragraphs = article['paragraphs']
        title = article['title']
        paragraphs_new = []
        for paragraph in paragraphs:
            flag = bernoulli.rvs(sample_rate)
            if flag == 1:
                paragraphs_new.append(paragraph)
        new_articles.append({'title': title, 'paragraphs': paragraphs_new})
    data = {'data': new_articles}
    with open(output_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    org_file = 'orig/dev.jsonl'
    new_file = 'qasrl.dev.json'
    change_Large_QA_SRL_SQuAD(org_file, new_file)
    # input_file = 'qasrl.train.json'
    # output_file = 'qasrl.train.sample.51K.json'
    # sample_data(input_file, output_file, 0.234)


