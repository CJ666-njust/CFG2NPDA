# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd

from collections import defaultdict


class GetGreibach(object):
    # init elements
    def __init__(self, data_dir: str, save_dir: str):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.N_list = None
        self.T_list = None
        self.P_list = None
        self.New_P_List = None
        self.S_list = None
        self.get_data()
        self.N_ = []  # 记录能够推出空的非终结符
        self.flag = True  # 记录删除空符何时结束
        self.pairN = []  # 记录所有单元偶对，列表元素为（A,B）类型
        self.reach_list = []  # 可达符号集合
        self.product_list = []  # 生成符号集合
        self.n_number = 0  # 当前使用的新非终结符序号，如T0，T1,...S

        # PDA的构造
        self.Q_PDA = ['q0', 'q1', 'qf']  # 此处的下推自动机可以只需要一个状态
        self.T_PDA = []  # 记录可以输入的非终结符，即greibach范式的N
        self.L_PDA = []  # 记录栈内可能的所有符号，即greibach范式的NvT
        self.move_PDA = []  # 记录所有的状态转换函数
        # (q, a, A) = (q1, aA) 表示当前状态为q, 输入a，栈顶为A，则进入状态q1, 用aA替换a

        self.q0_PDA = 'q0'  # 起始状态
        self.z0_PDA = 'z'  # 栈的开始符号
        self.F_PDA = 'qf'  # 表示空栈接受

        # 进行语言的推导
        self.memory_list = []  # 记录搜索过程中的分支状态。形如(state, 当前读入字符串索引, 当前stack, 对应len(move)-2)
        self.langua = ''

    # 获取N,T,P,S。@表示空；
    def get_data(self):
        assert os.path.exists(self.data_dir), "dir {} is not exist.".format(self.data_dir)
        data = pd.read_csv(self.data_dir, header=None)
        assert len(data), "txt is empty."
        N_list = []
        T_list = []
        P_list = [[] for _ in range(3, len(data))]
        S_list = []
        array = data.values[0::, 0::]  # 将读取的txt转成array类型
        # print("array:", array[0])
        # 分别读入V, T, P, S
        for v in array[0][0].split(' '):
            N_list.append(v)
        for t in array[1][0].split(' '):
            T_list.append(t)
        for s in array[2][0].split(' '):
            S_list.append(s)
        for i, x in enumerate(array[3:, :]):
            V, T = x[0].split("->")
            assert len(V) > 1, "产生式前继长度大于1，不是二型文法."
            # print(V)
            # print(T)
            temp = T.split("|")
            # print(temp)
            P_list[i].append(V.strip())
            # for ii, t in enumerate(temp)
            for k in temp:
                P_list[i].append(k)
        self.N_list = N_list
        self.T_list = T_list
        self.P_list = P_list
        self.S_list = S_list

    # 保存greibach范式
    def save_greibach(self):
        with open(self.save_dir + "/greibach.txt", 'w+') as f:
            f.write("非终结符集合：" + '\n')
            f.write(str(self.N_list) + '\n')
            f.write("终结符集合：" + '\n')
            f.write(str(self.T_list) + '\n')
            f.write("开始符号：" + '\n')
            f.write(str(self.S_list) + '\n')

            f.write("产生式集合：" + '\n')
            for x in self.P_list:
                f.write(str(x) + '\n')

    # 保存NPDA
    def save_npda(self):
        with open(self.save_dir + "/npda.txt", 'w+') as f:
            f.write("PDA的状态集合：" + '\n')
            f.write(str(self.Q_PDA) + '\n')
            f.write("PDA的输入字母表：" + '\n')
            f.write(str(self.T_PDA) + '\n')
            f.write("栈字母表：" + '\n')
            f.write(str(self.L_PDA) + '\n')
            f.write("转移函数集合：" + '\n')
            for x in self.move_PDA:
                f.write(str(x) + '\n')
            f.write("初始状态：" + '\n')
            f.write(str(self.q0_PDA) + '\n')
            f.write("栈的开始符号：" + '\n')
            f.write(str(self.z0_PDA) + '\n')
            f.write("终止状态：" + '\n')
            f.write(str(self.F_PDA) + '\n')

    # 找到列表中的重复元素，并列出索引。返回（n, [idx1, idx2]）,其中n为重复的元素值，idx为该元素的索引
    @staticmethod
    def find_repeat_elements(seq: list):
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return ((key, locs) for key, locs in tally.items()
                if len(locs) > 1)

    # 合并前继相同的产生式
    def combine_p(self):
        # print(self.P_list)
        lead_n = [self.P_list[i][0] for i in range(0, len(self.P_list))]
        need_delete = []
        for (n, index_list) in self.find_repeat_elements(lead_n):
            for x in index_list[1:]:
                need_delete.append(x)
            temp_list = []
            for i in range(1, len(index_list)):
                for p in self.P_list[index_list[i]][1:]:
                    # print('p:', p)
                    temp_list.append(p)
            for x in temp_list:
                self.P_list[index_list[0]].append(x)
            # print("need_delete:", need_delete)
        for p in need_delete:
            del self.P_list[p]
        # 消去产生式内重复元素
        for i in range(0, len(self.P_list)):
            self.P_list[i] = self.unique_1d_list(self.P_list[i])
        # print(self.P_list)

    # a为原字符串，b为需要从a中删去的字符串
    @staticmethod
    def replace_word(real_str: str, delete_str: str = ''):
        temp = ''
        c = real_str.split(delete_str)
        for word in c:
            temp += word
        return temp

    # 遍历所有产生式，若有生成 @， 放入N_,并删除后继 @
    def search_v(self):
        self.flag = False
        for p in self.P_list:
            for i, word in enumerate(p):
                if word is '@' or word is '':
                    self.N_.append(p[0])
                    self.flag = True
                    p[i] = '$'  # 该符号用来占位，使更改后的list和原list的shape相同，便于恢复非空的后继

    # 遍历所有产生式，若后继存在N_中元素，从后继中删去该元素
    def search_n(self):
        self.flag = False
        for ii, p in enumerate(self.P_list):
            for i, word in enumerate(p[1:]):
                for n_ in self.N_:
                    if n_ in word:
                        # print("{}, {}".format(n_, word))
                        self.P_list[ii][i + 1] = self.replace_word(word, n_)
                        # print(self.P_list[ii][i+1])
                        self.flag = True

    # 从P_list备份恢复不为空的后继,加回到产生式中
    def recover(self, p_copy):
        # print(p_copy)
        for i in range(0, len(p_copy)):
            for j in range(0, len(p_copy[i])):
                if self.P_list[i][j] == '$':
                    # print(1)
                    del p_copy[i][j]
                    del self.P_list[i][j]
        # print(p_copy)
        for i in range(0, len(p_copy)):
            for j in range(0, len(p_copy[i])):
                if p_copy[i][j] not in self.P_list[i]:
                    self.P_list[i].append(p_copy[i][j])

    """
    消去空产生式：
    将纯产生ϵ，如A->ϵ的推导规则消去;
    可致空符号：经过n步推导可以纯产生ϵ;
    第一步：找出所有可致空符号，获取N’;
    第二步：对部分推导规则进行转化，获取P1;
    第三步：考察起始非终结符S，获取N1和S1;
    第四步：G=(N1, T, P1, S1).
    """

    def delete_epsilon(self):
        P_copy = copy.deepcopy(self.P_list)
        while self.flag is True:
            self.search_v()
            self.search_n()

        self.recover(P_copy)
        # 如果S能直接推出空，则对N,P,S集合更新如下
        if 'S' in self.N_:
            self.N_list.append('S1')
            self.S_list.remove('S')
            self.S_list.append('S1')
            self.P_list.append(['S1', '@', 'S'])
        elif '@' in self.T_list:
            self.T_list.remove('@')

        # 删除长度为1的，即只能推出空的产生式,并将对应的非终结符从集合中删除
        for i, p in enumerate(self.P_list):
            if len(p) == 1:
                self.N_list.remove(p[0])
                del self.P_list[i]

        print('清除空产生式结束。')
        print("当前非终结符集合：", self.N_list)
        print("当前终结符集合：", self.T_list)
        print("当前产生式集合：", self.P_list)
        print("当前开始符号集合：", self.S_list)
        print("__" * 15)

    # 获取单元偶对
    def get_pair(self):
        for p in self.P_list:
            for x in p[1:]:
                if x in self.N_list:
                    self.pairN.append((p[0], x))

    # 删去二维列表中重复元素
    @staticmethod
    def unique_2d_list(seq: list):
        save_list = []
        for p in seq:
            temp_list = []
            for word in p:
                if temp_list.count(word) < 1:
                    temp_list.append(word)
            save_list.append(temp_list)
        return save_list

    # 删去一维列表重复元素
    @staticmethod
    def unique_1d_list(seq: list):
        save_list = []
        for x in seq:
            if save_list.count(x) < 1:
                save_list.append(x)
        return save_list

    # 遍历一次产生式集合，并根据单元偶对改变
    def search_one(self):
        print("遍历前：", self.P_list)
        P_copy = copy.deepcopy(self.P_list)
        lead_n = [self.P_list[i][0] for i in range(0, len(self.P_list))]
        for (first, second) in self.pairN:
            idx_1 = lead_n.index(first)
            idx_2 = lead_n.index(second)
            if second in self.P_list[idx_1]:
                # print(second)
                # 删除被替代的终结符
                idx = self.P_list[idx_1].index(second)
                del self.P_list[idx_1][idx]

                for p in self.P_list[idx_2]:
                    self.P_list[idx_1].append(p)
                # 终结符被重新引入，删去
                idx = self.P_list[idx_1].index(second)
                del self.P_list[idx_1][idx]
            # self.P_list[idx_1] = np.unique(self.P_list[idx_1]).tolist()
        self.P_list = self.unique_2d_list(self.P_list)
        print("遍历后：", self.P_list)
        print("__" * 10)
        if P_copy == self.P_list:
            return 0
        else:
            return 1

    """
    消去单产生式:
    第一步：根据当前产生式，有形如A->B，将（A,B）作为单元偶对；
    第二步：根据当前单元偶对，找到A->B产生式，将B更换为B为前继的产生式的后继，并删除后继中的重复元素；
    第三步：根据当前产生式，更新单元偶对；
    遍历第2，3步，直至产生式无变化后结束。
    """

    def delete_single_p(self):
        self.get_pair()
        print("初始单元偶对：", self.pairN)
        i = 1
        count = 0
        while i:
            count += 1
            print("进行第{}次遍历.".format(count))
            i = self.search_one()
            self.get_pair()
            print("当前单元偶对：", self.pairN)
        print("结束消单产生式遍历.共遍历{}次.".format(count))

    # 获取生成符号集合
    def get_product(self):
        product_list = []
        # 所有终结符都是生成符号
        for i, t in enumerate(self.T_list):
            product_list.append(t)
        flag = 1
        while flag:
            temp_list = copy.deepcopy(product_list)
            for p in self.P_list:
                for word in p[1:]:
                    count = 0
                    for letter in list(word):
                        if letter in product_list:
                            count += 1
                    if count == len(word):  # 若产生式的所有后继都属于生成符号，则前继也是生成符号
                        product_list.append(p[0])

            product_list = self.unique_1d_list(product_list)

            if temp_list == product_list:
                flag = 0

        print("__" * 30)
        print("获取的生成符号集合：", product_list)

        self.product_list = product_list

    # 获取可达符号集合
    def get_reach(self):
        reach_list = [s for s in self.S_list]  # 开始符号是可达的
        # 若产生式的前继是可达的，则产生式中所有的后继都是可达的
        flag = 1
        while flag:
            temp_list = copy.deepcopy(reach_list)
            for p in self.P_list:
                if p[0] in reach_list:
                    for word in p[1:]:
                        for letter in list(word):
                            reach_list.append(letter)
            reach_list = self.unique_1d_list(reach_list)

            if temp_list == reach_list:
                flag = 0

        reach_list.sort()
        print("获取的可达符号集合：", reach_list)
        self.reach_list = reach_list

    """
    消去无用符号
    有用符号=生成符号+可达符号，有用符号之外的符号就是无用符号;
    第一步：找出所有生成符号——即从生成字符串归约到S遇到的所有非终结符，
    除了这些非终结符之外，其它的所有非终结符都是非生成符号.将包含有这些非生成符号的推导规则全部删除;
    第二步：找出所有可达符号——即从S推导出到纯字符串遇到的所有非终结符;
    """

    # 必须先删除非生成符号，再删除不可达符号
    def delete_useless(self):
        # 获取生成符号集合
        G.get_product()
        # 先删除非生成符号
        not_product_list = [n for n in self.N_list]
        for t in self.T_list:
            not_product_list.append(t)

        not_product_list = [item for item in not_product_list if item not in set(self.product_list)]
        print("非生成符号集合：", not_product_list)
        for x in not_product_list:
            if x in self.N_list:
                self.N_list.remove(x)
            elif x in self.T_list:
                self.T_list.remove(x)

        # 删除包含非生成符号的产生式
        not_reach_list = [n for n in self.N_list]
        for t in self.T_list:
            not_reach_list.append(t)
        for t, p in enumerate(self.P_list):
            for i, word in enumerate(p):
                for ii, letter in enumerate(list(word)):
                    if letter in not_product_list:
                        if ii == 0:
                            del self.P_list[t]
                        else:
                            del self.P_list[t][i]

        print("__" * 30)
        # 获取可达符号集合
        G.get_reach()
        # 删除非可达符号
        not_reach_list = [item for item in not_reach_list if item not in set(self.reach_list)]
        print("非可达符号集合：", not_reach_list)
        for x in not_reach_list:
            if x in self.N_list:
                self.N_list.remove(x)
            elif x in self.T_list:
                self.T_list.remove(x)

        # 删除包含非生成符号的产生式
        for t, p in enumerate(self.P_list):
            for i, word in enumerate(p):
                for ii, letter in enumerate(list(word)):
                    if letter in not_reach_list:
                        if ii == 0:
                            del self.P_list[t]
                        else:
                            del self.P_list[t][i]
        print("__" * 30)
        print("消去无用符号结束.")
        print("当前非终结符集合：", self.N_list)
        print("当前终结符集合：", self.T_list)
        print("当前产生式集合：", self.P_list)
        print("__" * 30)

    """
    消除直接左递归：形如A→Aa。
    对于A→Aa|b（b可为空）。因为推导结束后一定有个b在开始位置，故改为：A→bB,B→aB。
    """

    # 消去直接左递归
    def delete_dirt_left_p(self):
        for i, p in enumerate(self.P_list):
            left_digui_idx = []
            left_digui = []
            not_left_idx = []
            for ii, word in enumerate(p[1:]):
                if word[0] == p[0]:
                    left_digui.append(word)
                    left_digui_idx.append(ii + 1)
                else:
                    not_left_idx.append(word)
            if len(left_digui_idx) == 0:
                continue
            # 举例：A->Aa1|Aa2|b1|b2
            new_v = p[0] + '1'  # 每次产生新符号在原符号后拼接1
            new_p1 = [p[0]]
            for word in not_left_idx:  # 添加产生式：A->b1|b2|b1A1|b2A1
                new_p1.append(word)
                new_p1.append(word + new_v)
            self.P_list.append(new_p1)
            self.N_list.append(new_v)

            new_p2 = [new_v]
            for ii, word in enumerate(left_digui):  # 添加产生式A1->a1|a2|a1A1|a2A1
                new_p2.append(word[1:])
                new_p2.append(word[1:] + new_v)
            self.P_list.append(new_p2)

            for ii in left_digui_idx:  # 删去原产生式中的左递归后继
                del self.P_list[i][ii]
            if len(self.P_list[i]) == 1:  # 如果原产生式就这么一个后继了，整条删去
                del self.P_list[i]
        # 合并相同前继产生式
        G.combine_p()

    # 消除全部左递归
    def delete_all_left_p(self):
        smaller_n_list = []
        for i, p in enumerate(self.P_list):
            delete_need = []
            for ii, word in enumerate(p[1:]):
                if word[0] in smaller_n_list:
                    delete_need.append(ii + 1)
                    idx = smaller_n_list.index(word[0])
                    # 添加新的后继
                    for x in self.P_list[idx][1:]:
                        self.P_list[i].append(x + word[1:])
            if len(delete_need) is not 0:
                for t in delete_need:
                    del self.P_list[i][t]
            smaller_n_list.append(p[0])
            self.delete_dirt_left_p()

        print("__" * 30)
        print("消除全部左递归结束。")
        print("当前产生式：", self.P_list)
        print("当前非终结符集合：", self.N_list)

    """
    修改为greibach范式：A-> aB1B2B3 或 A-> a 或 S-> @ 三种可能
    对于以非终结符开头的后继，将非终结符替换为对应的后继，直至为终结符号开头；
    对于不处于开头的终结符，引入新的非终结符和产生式，形如A1-> a;
    """

    def change_to_greibach(self):
        # 处理以非终结符开头的后继
        flag = 1
        while flag:
            p_copy = copy.deepcopy(self.P_list)
            for i, p in enumerate(self.P_list):
                delete_list = []
                for ii, word in enumerate(p[1:]):
                    if word[0] in self.N_list:
                        idx = self.N_list.index(word[0])
                        delete_list.append(ii + 1)
                        for x in self.P_list[idx][1:]:
                            self.P_list[i].append(x + word[1:])
                if len(delete_list) is not 0:
                    for t in delete_list:
                        del self.P_list[i][t]
            if p_copy == self.P_list:
                flag = 0
        print("当前产生式：", self.P_list)

        # 处理不处于后继开头的终结符
        new_t_list = []  # 记录需要生成新产生式的终结符
        for i, p in enumerate(self.P_list):
            for ii, word in enumerate(p):
                temp = list(word)
                for iii, letter in enumerate(list(word[1:])):
                    if letter in self.T_list and letter in new_t_list:
                        idx = new_t_list.index(letter)
                        new_n = 'T' + str(idx)

                        temp[iii + 1] = new_n
                        new_word = ''
                        for x in temp:
                            new_word += x
                        self.P_list[i][ii] = new_word
                    elif letter in self.T_list and letter not in new_t_list:
                        new_t_list.append(letter)
                        new_n = 'T' + str(self.n_number)
                        self.N_list.append(new_n)
                        self.n_number += 1
                        # temp = list(word)
                        temp[iii + 1] = new_n
                        new_word = ''
                        for x in temp:
                            new_word += x
                        self.P_list[i][ii] = new_word
        # print('new:', new_t_list)
        for i, x in enumerate(new_t_list):
            new_n = 'T' + str(i)
            new_p = [new_n, x]
            self.P_list.append(new_p)

        print("__" * 30)
        print("转换为greibach范式结束。")
        print("当前产生式：", self.P_list)
        print("当前非终结符集合：", self.N_list)

    """
    由greibach范式构造下推自动机PDAM
    """

    # 根据greibach范式初始化npda
    def npda_init(self):
        self.T_PDA = self.T_list  # 记录可以输入的非终结符，即greibach范式的N
        self.L_PDA = self.N_list  # 记录栈内可能的所有符号，即greibach范式的NvT
        # for x in self.T_list:
        # self.L_PDA.append(x)
        self.T_PDA.append('')
        self.L_PDA.append('z')  # 栈的开始符号

        # 引入开始转移和结束转移
        begin_move = [('q0', '', 'z'), ('q1', self.S_list[0] + 'z')]
        end_move = [('q1', '', 'z'), ('qf', 'z')]
        self.move_PDA.append(begin_move)
        self.move_PDA.append(end_move)

        # # PDA的构造
        # self.Q_PDA = ['q0', 'q1', 'qf']  # 此处的下推自动机可以只需要一个状态
        # self.T_PDA = []  # 记录可以输入的非终结符，即greibach范式的N
        # self.L_PDA = []  # 记录栈内可能的所有符号，即greibach范式的NvT
        # self.move_PDA = []  # 记录所有的状态转换函数
        # # (q, a, A) = (q1, aA) 表示当前状态为q, 输入a，栈顶为A，则进入状态q1, 用aA替换a
        #
        # self.q0_PDA = 'q0'  # 起始状态
        # self.z0_PDA = 'z'  # 栈的开始符号
        # self.F_PDA = 'qf'  # 表示空栈接受

    # 合并相同前继的转移函数
    def combine_pda(self):
        delete_idx = []
        temp_lead = None
        temp_lead_idx = -1
        count = 0
        for i in range(0, len(self.move_PDA)):
            if self.move_PDA[i][0] == temp_lead:
                self.move_PDA[temp_lead_idx].append(self.move_PDA[i][1])
                self.move_PDA[i] = '@@@'
                count += 1
            elif self.move_PDA[i] is not temp_lead:
                temp_lead = self.move_PDA[i][0]
                temp_lead_idx = i
        for i in range(0, count):
            self.move_PDA.remove("@@@")

    # 获取文法的npda
    def get_npda(self):
        self.npda_init()
        for i, p in enumerate(self.P_list):
            for ii, word in enumerate(p[1:]):
                if len(word) > 1:
                    move = [('q1', word[0], p[0]), ('q1', word[1:])]
                    self.move_PDA.append(move)
                else:
                    move = [('q1', word[0], p[0]), ('q1', '')]
                    self.move_PDA.append(move)
        self.combine_pda()
        print("__" * 30)
        print("获取npda结束。")
        print("PDA的状态集合：", self.Q_PDA)
        print("PDA的输入字母表：", self.T_PDA)
        print("栈字母表：", self.L_PDA)
        print("转移函数集合：", self.move_PDA)
        print("初始状态：", self.q0_PDA)
        print("栈的开始符号：", self.z0_PDA)
        print("终止状态：", self.F_PDA)

    # 逐步执行
    def forward(self):
        # 合并相同前继产生式
        self.combine_p()
        # 消去空产生式
        self.delete_epsilon()

        # 消去单产生式
        self.delete_single_p()
        # 消去无用产生式
        self.delete_useless()
        # 消去直接左递归
        self.delete_dirt_left_p()
        print("__" * 30)
        print("消去直接左递归结束。")
        print("当前产生式：", G.P_list)
        print("当前非终结符集合：", G.N_list)
        # 消去全部左递归
        self.delete_all_left_p()

        # 消去空产生式
        self.delete_epsilon()
        # 消去无用产生式
        self.delete_useless()

        # 构造Greibach范式
        self.change_to_greibach()
        # 保存Greibach范式
        self.save_greibach()
        # 获取NPDA
        self.get_npda()
        # 保存NPDA
        self.save_npda()

    # 读入需要判断的语言
    def get_language(self, langua_dir):
        data = pd.read_csv(langua_dir, header=None)
        # print(data.values)
        self.langua = data.values[0][0]
        return data.values[0][0]

    # 判断是否符合文法规则
    def is_conform(self, language: str, save_log, stack=None, state: str = 'q1'):
        if stack is None:
            stack = ['z', 'S']
        state = 'q1'
        len_str = len(language)
        language = list(language)

        for i, letter in enumerate(language):
            with open(save_log, 'a', encoding='utf-8') as f:
                f.write("当前处理语句:" + str(language) + '\n')
                f.write("当前处理的字母索引:" + str(i) + '\n')
                f.write("当前stack:" + str(stack) + '\n')
                f.write("当前state:" + state + '\n')
                f.write("__"*10 + '\n')

            # print("__" * 30)
            # print("当前已处理字符：", language[0:i])
            for ii, p in enumerate(self.move_PDA[2:]):

                print("now state:", (state, letter, stack[-1]))
                print("p[0]:", p[0])
                if p[0] == (state, letter, stack[-1]) and len(p) > 2:  # 有匹配的转移函数，且有多个后继
                    print((state, letter, stack[-1]))
                    last_idx = 2 + ii  # 记录最近回溯点的转移函数序号
                    print("进入条件1：", p[-1])
                    temp = copy.deepcopy(stack)
                    self.memory_list.append((state, i, temp, len(p) - 2, last_idx))
                    new_q, new_str = p[-1]
                    state = new_q
                    stack.pop()
                    # print(new_str)
                    new_str = list(new_str)
                    new_str.reverse()
                    temp_x = ''
                    for x in new_str[0:]:
                        if '0' <= x <= '9':
                            temp_x += x
                        else:
                            if temp_x is not '':
                                stack.append(x + temp_x)
                                temp_x = ''
                            else:
                                stack.append(x)
                    print("stack:", stack)
                    if stack == ['z'] and letter == language[-1]:
                        return 1
                    elif stack == ['z'] and letter is not language[-1]:
                        return 0
                    elif stack is not ['z'] and i == len(language) - 1:
                        return 0
                    # elif stack is not ['z'] and letter == language[-1]:
                    #     return 0
                    break

                elif p[0] == (state, letter, stack[-1]) and len(p) == 2:  # 有匹配的转移函数，仅一个后继
                    new_q, new_str = p[-1]
                    state = new_q
                    stack.pop()
                    new_str = list(new_str)
                    new_str.reverse()
                    temp_x = ''
                    for x in new_str[0:]:
                        if '0' <= x <= '9':
                            temp_x += x
                        else:
                            if temp_x is not '':
                                stack.append(x + temp_x)
                                temp_x = ''
                            else:
                                stack.append(x)
                    if stack == ['z'] and letter == language[-1]:
                        return 1
                    elif stack == ['z'] and letter is not language[-1]:
                        return 0

                    break

                elif (p[0] is not (state, letter, stack[-1]) and
                      ii == (len(self.move_PDA) - 3) and
                      len(self.memory_list) > 0) or ((ii == len(self.move_PDA) - 3) and
                                                     len(self.memory_list) > 0 and i == len_str - 1):  # 无匹配或字符串末尾，回溯
                    (load_state, load_idx, load_stack, load_num, last_idx) = self.memory_list.pop()
                    if load_num > 1:
                        self.memory_list.append((load_state, load_idx, load_stack, load_num - 1, last_idx))
                    temp_str = ''
                    for x in self.langua[load_idx:]:
                        temp_str += x
                    with open(save_log, 'a', encoding='utf-8') as f:
                        f.write("发生回溯。" + '\n')
                        f.write("回溯到状态：" + str((load_state, load_idx, load_stack, load_num, last_idx)) + '\n')
                        f.write("__"*20 + '\n')
                    print("回溯了。")
                    # 回溯后的第一步操作已固定；
                    new_q1, new_str1 = self.move_PDA[last_idx][load_num]
                    load_state = new_q1
                    new_str1 = list(new_str1)
                    load_stack.pop()
                    new_str1.reverse()
                    temp_x = ''
                    for x in new_str1[0:]:
                        if '0' <= x <= '9':
                            temp_x += x
                        else:
                            if temp_x is not '':
                                load_stack.append(x + temp_x)
                                temp_x = ''
                            else:
                                load_stack.append(x)
                    return self.is_conform(temp_str[1:], save_log, load_stack, load_state)


if __name__ == '__main__':
    text_dir = "./datas.txt"
    save_dir = "./save"
    language_dir = "./language/3.txt"
    # 读入二型文法
    G = GetGreibach(text_dir, save_dir)
    # 执行生成greibach和 npda
    G.forward()
    # 读取语言
    langua = G.get_language(language_dir)
    # 判断该语言是否属于该文法
    save_log_dir = "./log" + language_dir[1:]
    print('__' * 30)
    print("该语言属于该文法吗：", G.is_conform(langua, save_log_dir))
