import csv
import json
import re
import enchant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np
import joblib
import matplotlib.pyplot as plt
# import language_tool_python as checker


special_character = [',', ';', '.']

special_word = ['ORGANIZATION', 'CAPS', 'DATE']
transition_words= ['accordingly', 'as a result', 'consequently', 'since', 'for that reason', 'therefore',
                   'and so', 'hence', 'thus', 'because', 'on account of', 'after', 'afterwards', 'always', 'never',
                   'later', 'soon', 'subsequently', 'at length', 'this time', 'simultaneously', 'until now', 'during',
                   'earlier', 'following', 'whenever', 'so far', 'immediately', 'in the mean time', 'furthermore',
                   'however', 'as a result', 'after all', 'on the contrary', 'on the other hand', 'in contrast',
                   'otherwise', 'at the same time', 'nonetheless', 'nevertheless', 'likewise', 'wherever',
                   'in other words', 'in conclusion', 'in the end', 'to conclude', 'to summarize', 'in summary',
                   'in short', 'to sum up'
                   ]


def csv2json(csv_path, json_path):
    data = []
    with open(csv_path, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            rater1_domain1 = int(rows["rater1_domain1"]) if rows["rater1_domain1"] else 0
            rater2_domain1 = int(rows["rater2_domain1"]) if rows["rater2_domain1"] else 0
            domain1_score = int(rows["domain1_score"]) if rows["domain1_score"] else 0
            data.append({"essay_set": int(rows["essay_set"]), "essay": rows["essay"],
                         "rater1_domain1": rater1_domain1, "rater2_domain1": rater2_domain1,
                         "domain1_score": domain1_score})
    with open(json_path, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))


def normalize_array(array):
    norm = np.linalg.norm(array)
    normal_array = array / norm
    return normal_array


def count_transition_words(essay):
    word_count = 0
    for word in transition_words:
        if word in essay:
            word_count += 1
    return word_count


def feature_vector(data):
    essay = data['essay']
    # print(essay)
    essay_array = re.split('([^a-zA-Z0-9])', essay)
    checker = enchant.Dict("en_US")
    nof_comma, nof_period, nof_semicolon, nof_exclamation, nof_spelling_mistake, sentence_word_count = 0, 0, 0, 0, 0, 0
    newline_count = 0
    word_length_array = []
    sentence_length_array = []
    para_length_array = []
    para_word_count = 0
    # print(essay_array)

    for segment in essay_array:
        if len(segment) == 1:
            if segment == '.':
                nof_period = nof_period + 1
                sentence_length_array.append(sentence_word_count)
                sentence_word_count = 0
            if segment == '!':
                nof_exclamation = nof_exclamation + 1
                sentence_length_array.append(sentence_word_count)
                sentence_word_count = 0
            elif segment == ',':
                nof_comma = nof_comma + 1
            elif segment == ';':
                nof_semicolon = nof_semicolon + 1
            elif segment == 'I' or segment == 'A' or segment == 'a':
                word_length_array.append(len(segment))
                sentence_word_count = sentence_word_count + 1
            elif segment == '\n':
                para_length_array.append(para_word_count)
                newline_count += 1
        elif len(segment) > 1:
            para_word_count += 1
            word_length_array.append(len(segment))
            if not any(ele in segment for ele in special_word) and not checker.check(segment):
                nof_spelling_mistake = nof_spelling_mistake + 1
            sentence_word_count = sentence_word_count + 1
    if len(sentence_length_array) == 0:
        sentence_length_array.append(sentence_word_count)
    word_length_array = normalize_array(word_length_array)
    sentence_length_array = normalize_array(sentence_length_array)
    return [np.amax(word_length_array), np.amin(word_length_array), np.std(word_length_array),
            nof_spelling_mistake,
            np.std(sentence_length_array), np.amax(sentence_length_array), np.amin(sentence_length_array),
            nof_comma, nof_period, nof_semicolon, nof_exclamation,
            count_transition_words(essay), data['essay_set']]
        #, np.mean(para_length_array), np.std(para_length_array)]


def data_summarize(data):
    essay_data = []
    essay_lables = []
    for value in data:
        print(data.index(value))
        essay_data.append(feature_vector(value))
        essay_lables.append(value['rater1_domain1'])
    return {"student_data": essay_data, "student_labels": essay_lables}


def LogisticRegressionModel(data):
    x_train, x_test, y_train, y_test = train_test_split(data["student_data"], data["student_labels"], shuffle=True,
                                                        test_size=0.33)
    # print("I am here")
    # print(x_train)
    # print(y_train)
    # logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logreg = SVR(kernel='linear')
    logreg.fit(x_train, y_train)
    print('Finished Training')
    PATH = './svr_essay.pth'
    s = joblib.dump(logreg,PATH)
    test = joblib.load(PATH)
    y_pred = test.predict(x_test)
    # print("-------")
    # print(y_test)
    # print(y_pred)
    score = test.score(x_test, y_test)
    print("R-squared:", score)
    print("MSE:", mean_squared_error(y_test, y_pred))
    # print("X: ",len(x_test))
    # print(x_test)
    # print("Y: ",len(y_test))
    # print(y_test)
    # print(y_pred)
    print(logreg.coef_)
    plt.scatter(y_test, y_pred, s=5, color="blue", label="original")

    plt.show()

if __name__ == "__main__":
    json_path = 'data/training_set_rel3.json'
    # csv2json('data/training_set_rel3.csv','data/training_set_rel3.json')
    f = open(json_path)
    test_data = json.load(f)
    # all_data = data_summarize(test_data[-1000:])
    # LogisticRegressionModel(all_data)
    LogisticRegressionModel(data_summarize(test_data))
