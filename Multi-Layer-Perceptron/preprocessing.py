import random
from operator import itemgetter
import os


def process_files():
    entries = []

    with open("letter-recognition.txt") as f:
        for line in f:
            entries.append(line.replace('\n', '').split(','))
        f.close()

    entries = sorted(entries, key=itemgetter(0))
    # normalize data
    for j in range(1, len(entries[0])):
        col_max = int(entries[0][j])
        col_min = int(entries[0][j])
        for i in range(0, len(entries)):
            entries[i][j] = int(entries[i][j])
            if entries[i][j] > col_max:
                col_max = entries[i][j]
            if entries[i][j] < col_min:
                col_min = entries[i][j]
        for i in range(0, len(entries)):
            entries[i][j] = (int(entries[i][j]) - col_min) / (col_max - col_min)

    # 2/3 for training, 1/3 for testing
    training_list = []
    testing_list = []
    i = 0
    while i < len(entries) - 2:
        training_list.append(entries[i])
        testing_list.append(entries[i + 1])
        training_list.append(entries[i + 2])
        testing_list.append(entries[i + 3])
        training_list.append(entries[i + 4])
        i += 5

    random.shuffle(training_list)
    random.shuffle(testing_list)

    training = open("training.txt", "w")
    testing = open("testing.txt", "w")

    for entry in training_list:
        training.write(str(entry)[1:-1].replace("'", "").replace(",", "") + "\n")

    for entry in testing_list:
        testing.write(str(entry)[1:-1].replace("'", "").replace(",", "") + "\n")

    remove_chars = len(os.linesep)
    training.truncate(training.tell() - remove_chars)
    testing.truncate(testing.tell() - remove_chars)

    training.close()
    testing.close()


def get_params(filename):
    parameters = []
    cnt = 0
    with open(filename) as f:
        for line in f:
            if cnt < 4:
                param = int(line[line.find(' ') + 1:-1])
            elif 6 > cnt >= 4:
                param = float(line[line.find(' ') + 1:-1])
            elif cnt == 6:
                param = int(line[line.find(' ') + 1:-1])
            elif cnt == 7:
                param = line[line.find(' ') + 1:-1]
            else:
                param = line[line.find(' ') + 1:]
            parameters.append(param)
            cnt += 1
    return parameters
