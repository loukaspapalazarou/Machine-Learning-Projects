import os
import random
import matplotlib.pyplot as plt

def get_parameters(file_name):
    cnt = 0
    params = []
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        if cnt < 3:
            params.append(int(line[line.find(' ') + 1:-1]))
        elif cnt == 3:
            l_rates = line[line.find(' ') + 1:-1].split(' ')
            for i in range(len(l_rates)):
                l_rates[i] = float(l_rates[i])
            params.append(l_rates)
        elif cnt == 4:
            params.append(float(line[line.find(' ') + 1:-1]))
        elif cnt == 5:
            params.append(int(line[line.find(' ') + 1:-1]))
        else:
            params.append(line[line.find(' ') + 1:].replace('\n', ''))
        cnt += 1
    f.close()
    return params

def process_data(input_file, train_file, test_file):
    # read data
    data = []
    f = open(input_file, "r")
    lines = f.readlines()
    lines = lines[3:-2]
    for line in lines:
        data.append(line.replace('\n', '').replace('<', '').replace('*','').split(','))
    f.close()

    # normalize data
    for column in range(1, len(data[0])):
        temp = []
        for line in range(len(data)):
            temp.append(float(data[line][column]))
        for line in range(len(data)):
            data[line][column] = float((float(data[line][column]) - min(temp)) / (max(temp) - min(temp)))

    # write data
    random.shuffle(data)
    train_file = open(train_file, "w+")
    test_file = open(test_file, "w+")
    norm =  open("normalized.txt", "w+")
    temp = 0
    for indx in range(len(data)):
        s = str(data[indx])[1:-1].replace("'", "").replace(",", "") + "\n"
        s = s[s.find(' ')+1:]
        norm.write(s)
        if temp == 2:
            test_file.write(s)
            temp = 0
        else:
            train_file.write(s)
            temp += 1

    remove_chars = len(os.linesep)
    train_file.truncate(train_file.tell() - remove_chars)
    test_file.truncate(test_file.tell() - remove_chars)

    train_file.close()
    test_file.close()
    norm.close()

def create_centers_file(filename, dimension, numOfNodes):
    f = open(filename, "w+")
    for i in  range(numOfNodes):
        for j in range(dimension):
            n = random.uniform(0.0, 1.0)
            f.write(str(n) + " ")
        f.write("\n")
    remove_chars = len(os.linesep)
    f.truncate(f.tell() - remove_chars)
    f.close()

def get_data_dimension(filename):
    f = open(filename, "r")
    l = f.readline()
    l = l.split(' ')
    f.close()
    return len(l)-1

def file_to_list(filename):
    c = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')[0:-1].split(' ')
        c.append([float(x) for x in line])
    f.close()
    return c

def save_errors(train_errors, test_errors, filename):
    f = open(filename, "w+")
    for i in range(len(train_errors)):
        f.write(str(i) + "\t" + str(format(train_errors[i], '.16f')) + "\t" + str(format(train_errors[i], '.16f')) + "\n")
    remove_chars = len(os.linesep)
    f.truncate(f.tell() - remove_chars)
    f.close()

def save_plot(train_errors, test_errors, filename):
    plt.plot(train_errors, label="Train error")
    plt.plot(test_errors, label="Test error")
    plt.legend()
    plt.savefig(filename)

def save_weights(weights, filename):
    f = open(filename, "w+")
    for w in weights:
        f.write(str(w) + "\n")
    remove_chars = len(os.linesep)
    f.truncate(f.tell() - remove_chars)
    f.close()