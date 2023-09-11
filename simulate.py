import random
import tenth_state
#action 0: 리셋, 1: 진행
#reward next_state로 나아가는데 드는 돈
#current_state: [현재 강화 단계, 성공 횟수(메인 스탯에 붙은 횟수)]
#objective 도달하고자 하는 성공 횟수
#처음 10단계는 action과 무관하게 행동이 같으므로, 그 과정을 생략

def step(current_state, action, objective):
    end_flag = 0
    reward = 0
    #prob: 성공 횟수별, 강화 시 성공 확률 ex) 0번 성공 상태시, 다음 스텝에서의 성공 확률 0.35
    prob = [0.35,0.35,0.35,0.2,0.2,0.2,0.2,0.15,0.1,0.05]
    # 각 강화 단계별 강화 비용 ex) 0번 성공 상태시, 강화 비용 10원
    reward_li = [-10, -10, -10, -20, -20, -20, -20, -30, -30, -50]
    #목표 강화 수치에 도달했을 경우 다음 강화에 들어가지 않고, 강화가 종료되었음을 알림
    if current_state[1] >= objective:
        next_state = current_state
        end_flag = 1
    # 목표 강화 수치에 도달하지 못했을 경우, 비용을 내고 강화에 도전함. 강화 시도 후의 state, reward(강화 비용), 강화가 종료되었는지의 여부를 알려줌(이 경우에는 종료되지 않음)
    else:
        if current_state[0]<=19:
            #action이 0일경우, 현 강화 단계를 초기화하고, 10회 강화 진행
            if action == 0:
                next_state, reward, end_flag = tenth_state.tenth_state_predict(objective)
            #action이 0이 아닐 경우, 강화 단계를 초기화하지 않고, 다음 강화를 진행
            else:
                reward = reward_li[current_state[1]]
                if prob[current_state[1]] >= random.random():
                    next_state = [current_state[0] + 1, current_state[1]+1]
                else:
                    next_state = [current_state[0] + 1, current_state[1]]
        #강화가 총 20회까지 진행이 가능하므로, 20회 강화를 했음에도 목표치에 도달하지 못했으면, 강화 단계를 초기화하고, 10회 강화 진행
        elif current_state[0] == 20:
            next_state, reward, end_flag = tenth_state.tenth_state_predict(objective)
        #잘못된 state가 들어왔을 경우 오류임을 알려줌
        else:
            print('error')
            return 0
    return [next_state,reward,end_flag]


# current_state = [20,7]
# action = 1
# objective = 8
# print(step(current_state, action, objective))

