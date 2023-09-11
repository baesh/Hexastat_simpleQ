import numpy as np
import random
import copy

import simulate

#목표로 하는 강화 성공 수치
objective = 8
#in this case, similar with learning rate
alpha = 0.000001
#강화를 할 것인지, 초기화 할 것인지의 2택이기 때문에, 각 state에서의 action별 Q table을 작성
#10회 강화하기 전까지는 강화 단계를 초기화할 수 없어 action 선택이 무의미하기 때문에, Q table은 강화가 10회~20회 진행되었을 때 까지만의 데이터만을 포함하면 됨.
#강화는 총 0회~objective회 성공할 수 있음. (objective회가 넘게 성공한 경우는, objective회 강화가 성공했을 때 강화를 더이상 진행하지 않기 때문에 고려할 필요가 없음)
#따라서, Q table은 2*11*(objective+1)의 사이즈를 가지면 됨.
#편의를 위해, 강화를 초기화 할 때의 Q value를 담은 Q_table_action_zero, 강화를 계속 진행할 경우의 Q value를 담은 Q_table_action_one으로 분류
#i.e. 같은 위치에 있는 Q_table_action_zero의 값이 Q_table_action_one값보다 클 경우 강화를 초기화, 반대의 경우 강화를 계속 진행
#ex) n번 강화를 시도하여 m번 강화에 성공한 state의 Q 값은, 강화를 초기화 하는 경우 Q_table_action_zero[n-10,m], 강화를 계속 진행하는 경우 Q_table_action_one[n-10,m]
Q_table_action_zero = np.ones([11,objective+1])
Q_table_action_one = np.ones([11,objective+1])

#사전에 학습을 진행하던 기록이 있으면, 이어서 학습을 진행
f0 = open('Q_table_action_zero.txt', 'r')
lines0 = f0.readlines()
f0_data = []
for line in lines0:
    line_li = line.split(',')
    line_li.pop()
    float_line_li = []
    for element in line_li:
        float_line_li.append(float(element))
    f0_data.append(float_line_li)
f0.close()
print(f0_data)
if len(f0_data) != 0:
    Q_table_action_zero = np.array(f0_data)

f1 = open('Q_table_action_one.txt', 'r')
lines1 = f1.readlines()
f1_data = []
for line in lines1:
    line_li = line.split(',')
    line_li.pop()
    float_line_li = []
    for element in line_li:
        float_line_li.append(float(element))
    f1_data.append(float_line_li)
f1.close()
print(f1_data)
if len(f1_data) != 0:
    Q_table_action_one = np.array(f1_data)

#학습 epoch
epoch_no = 10000000000

for epoch in range(epoch_no):
    # 현재 강화 상태를 10회~20회 진행 중 0회 ~ objective회 성공 중 하나로 랜덤하게 설정
    current_state = [random.randint(10,20), random.randint(0,objective)]

    # 강화 단계에 해당하는 Q table의 value를 보고 강화를 계속 진행할 것인지, 초기화 할 것인지 판단
    if Q_table_action_zero[current_state[0]-10,current_state[1]] >= Q_table_action_one[current_state[0]-10,current_state[1]]:
        action = 0
    else:
        action = 1
    # epsilon greedy하게 action을 선택
    if random.random() < 0.1:
        action = abs(action -1)
    
    #어떤 action을 취했냐에 따라, 현 상태의 action에 대한 Q 값을 Q_table_action_zero에서 가져올 것인지, Q_table_action_one에서 가져올 것인지 결정
    if action == 0:
        current_Q = copy.deepcopy(Q_table_action_zero[current_state[0]-10,current_state[1]])
    else:
        current_Q = copy.deepcopy(Q_table_action_one[current_state[0]-10,current_state[1]])

    #현재 상태에서 어떤 action을 취했냐에 따라 next state와 그에 상응하는 reward(이 경우 강화 비용)을 가져옴
    next_state,reward,end_flag = simulate.step(current_state,action,objective)

    # next_state에 대해서도, Q table을 보고, 어느 action이 최적일지 판단
    if Q_table_action_zero[next_state[0]-10,next_state[1]] > Q_table_action_one[next_state[0]-10,next_state[1]]:
        next_best_action = 0
    else:
        next_best_action = 1
    # next state의 Q value를, next state의 최적 action에 의거해 가져옴
    if next_best_action == 0:
        next_Q = copy.deepcopy(Q_table_action_zero[next_state[0]-10,next_state[1]])
    else:
        next_Q = copy.deepcopy(Q_table_action_one[next_state[0]-10,next_state[1]])

    # Q table을 reward + next_Q - current_Q 값에 의해 update
    if action == 0:
        Q_table_action_zero[current_state[0]-10,current_state[1]] = current_Q + alpha * (reward + next_Q - current_Q)
    else:
        Q_table_action_one[current_state[0] - 10, current_state[1]] = current_Q + alpha * (reward + next_Q - current_Q)


    # 100000번 루프가 돌 때마다, 각 state별로 강화를 하는 것이 좋은지, 초기화하는 것이 좋은지 알려주는 table인 action guide를 출력
    #ex) action_guide[m,n] = 0 => m+10번 강화를 시도했을 때, n번 성공했다면 강화를 더 진행하지 않고, 초기화한다.
    #ex) action guide[i,j] = 1 => i+10번 강화를 시도했을 때, j번 성공했다면, 강화를 초기화하지 않고, 계속 진행한다.
    if epoch%100000 == 0:
        
        action_guide = np.zeros([11,objective+1])
        for i in range(11):
            for j in range(objective+1):
                if Q_table_action_zero[i,j] < Q_table_action_one[i,j]:
                    action_guide[i,j] = 1

        print(action_guide)
        # print(Q_table_action_zero)
        # print(Q_table_action_one)
        
        #현재까지의 Q table을 저장
        f0 = open('Q_table_action_zero.txt', 'w')
        f1 = open('Q_table_action_one.txt','w')
        for i in range(11):
            for j in range(objective + 1):
                f0.write(str(Q_table_action_zero[i,j])+',')
                f1.write(str(Q_table_action_one[i, j])+',')
            f0.write('\n')
            f1.write('\n')
        f0.close()
        f1.close()

