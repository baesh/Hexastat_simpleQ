import random
#10회까지는 초기화가 불가능하기 때문에, action과 무관하게 10회째 강화할때까지의 state 예측 가능
#state가 [n,m]일 경우 n회 강화 시도하여 m회 강화에 성공한 상태
def step_before_tenth(current_state, objective):
    end_flag = 0
    #prob: 성공 횟수별, 강화 시 성공 확률 ex) 0번 성공 상태시, 다음 스텝에서의 성공 확률 0.35
    prob = [0.35,0.35,0.35,0.2,0.2,0.2,0.2,0.15,0.1,0.05]
    #각 강화 단계별 강화 비용 ex) 0번 성공 상태시, 강화 비용 10원
    reward_li = [-10,-10,-10,-20,-20,-20,-20,-30,-30,-50]
    #목표 강화 수치에 도달했을 경우 다음 강화에 들어가지 않고, 강화가 종료되었음을 알림
    if current_state[1] >= objective:
        next_state = current_state
        reward = 0
        end_flag = 1
    #목표 강화 수치에 도달하지 못했을 경우, 비용을 내고 강화에 도전함. 강화 시도 후의 state, reward(강화 비용), 강화가 종료되었는지의 여부를 알려줌(이 경우에는 종료되지 않음)
    else:
        reward = reward_li[current_state[1]]
        if prob[current_state[1]] >= random.random():
            next_state = [current_state[0] + 1, current_state[1]+1]
        else:
            next_state = [current_state[0] + 1, current_state[1]]

    return [next_state,reward,end_flag]

#10번째까지 강화 시, 어떤 상태에 도달했는지 알려줌
def tenth_state_predict(objective):
    #강화 시작 시점은 0번 시도, 0회 성공 상태
    initial_state = [0,0]
    current_state = initial_state
    #reward_sum: 10회 강화할때까지의 누적 강화 비용
    reward_sum = 0
    #10번 강화를 하되, 그 전에 목표치에 도달했을 경우, 사전 종료
    for i in range(10):
        current_state,reward,end_flag = step_before_tenth(current_state, objective)
        reward_sum = reward_sum + reward
        if end_flag == 1:
            break

    return [current_state, reward_sum, end_flag]

# current_state, reward_sum, end_flag = tenth_state_predict(8)
# print(current_state, reward_sum, end_flag)


