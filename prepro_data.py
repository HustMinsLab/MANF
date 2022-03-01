# -*- coding: utf-8 -*-

import re

class_lable = [['Expansion'], ['Comparison'], ['Contingency'], ['Temporal']]

def prepro_data_train(train_file_list):
    train_idx = []
    train_label = []
    train_arg_1 = []
    train_arg_2 = []
    train_conn = []
    train_label_list = []

    for line in train_file_list:
        train_idx.append(line[0].strip('\n').split(' '))
        train_label.append(line[4].strip('\n').split(' '))
        line[6] = re.sub(r'[^A-Za-z0-9 ]+', '', line[6])
        line[7] = re.sub(r'[^A-Za-z0-9 ]+', '', line[7])
        train_arg_1.append(line[6].strip('\n').split(' ')[0:50])
        train_arg_2.append(line[7].strip('\n').split(' ')[0:50])
        train_conn.append(line[8].strip('\n').split(' '))
        
    for cla in train_label:
        if cla == class_lable[0]:
            train_label_list.append([1,0,0,0])
        elif cla == class_lable[1]:
            train_label_list.append([0,1,0,0])
        elif cla == class_lable[2]:
            train_label_list.append([0,0,1,0])
        elif cla == class_lable[3]:
            train_label_list.append([0,0,0,1])
        else: print('error')  
    
    return train_arg_1, train_arg_2, train_label_list

def prepro_data_dev(dev_file_list):
    dev_idx = []
    dev_label_1 = []
    dev_label_2 = []
    dev_arg_1 = []
    dev_arg_2 = []
    dev_conn_1 = []
    dev_conn_2 = []
    dev_label_list = []

    for line in dev_file_list:
        dev_idx.append(line[0].strip('\n').split(' '))
        dev_label_1.append(line[4].strip('\n').split(' '))
        dev_label_2.append(line[5].strip('\n').split(' '))
        line[7] = re.sub(r'[^A-Za-z0-9 ]+', '', line[7])
        line[8] = re.sub(r'[^A-Za-z0-9 ]+', '', line[8])
        dev_arg_1.append(line[7].strip('\n').split(' ')[0:50])
        dev_arg_2.append(line[8].strip('\n').split(' ')[0:50])
        dev_conn_1.append(line[9].strip('\n').split(' '))
        dev_conn_2.append(line[11].strip('\n').split(' '))
    
    for cla in dev_label_1:
        if cla == class_lable[0]:
            dev_label_list.append([1,0,0,0])
        elif cla == class_lable[1]:
            dev_label_list.append([0,1,0,0])
        elif cla == class_lable[2]:
            dev_label_list.append([0,0,1,0])
        elif cla == class_lable[3]:
            dev_label_list.append([0,0,0,1])
        else: print('error') 
    
    return dev_arg_1, dev_arg_2, dev_label_list

def prepro_data_test(test_file_list):
    test_idx = []
    test_label_1 = []
    test_label_2 = []
    test_arg_1 = []
    test_arg_2 = []
    test_conn_1 = []
    test_conn_2 = []
    test_label_list = []
    
    for line in test_file_list:
        test_idx.append(line[0].strip('\n').split(' '))
        test_label_1.append(line[4].strip('\n').split(' '))
        test_label_2.append(line[5].strip('\n').split(' '))
        line[7] = re.sub(r'[^A-Za-z0-9 ]+', '', line[7])
        line[8] = re.sub(r'[^A-Za-z0-9 ]+', '', line[8])
        test_arg_1.append(line[7].strip('\n').split(' ')[0:50])
        test_arg_2.append(line[8].strip('\n').split(' ')[0:50])
        test_conn_1.append(line[9].strip('\n').split(' '))
        test_conn_2.append(line[11].strip('\n').split(' '))
    
    for cla in test_label_1:
        if cla == class_lable[0]:
            test_label_list.append([1,0,0,0])
        elif cla == class_lable[1]:
            test_label_list.append([0,1,0,0])
        elif cla == class_lable[2]:
            test_label_list.append([0,0,1,0])
        elif cla == class_lable[3]:
            test_label_list.append([0,0,0,1])
        else: print('error')
        
    return test_arg_1, test_arg_2, test_label_list
