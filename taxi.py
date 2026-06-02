# 필요한 라이브러리를 불러옵니다.
import gymnasium as gym
import numpy as np
import pandas as pd

# 학습을 완료한 에이전트를 시각화하는데 필요한 라이브러리
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Taxi-v4 환경 생성
env = gym.make('Taxi-v4')

# 상태(State)와 행동(Action) 개수 출력
print([env.observation_space.n, env.action_space.n])

# 학습 파라미터
learning_rate = 0.1
gamma = 0.95
num_episodes = 5000
max_steps = 100

# 랜덤 초기 Q-Table 생성
Q_1ep = np.random.rand(env.observation_space.n, env.action_space.n)

# 환경 초기화
state = env.reset()[0]

done = False

# 1 Episode 학습 예시
for i in range(max_steps):

    # 현재 상태에서 최대 Q값 행동 선택
    action = np.argmax(Q_1ep[state,:])

    # 행동 수행
    next_state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    # 오차 계산
    error = reward + gamma * np.max(Q_1ep[next_state,:]) - Q_1ep[state,action]

    # Q 업데이트
    Q_1ep[state,action] = (
        Q_1ep[state,action] + learning_rate * error
    )

    state = next_state

    if done:
        break


# Q-learning 함수 정의
def Q_learning(env, gamma, learning_rate, num_episodes, max_steps, Q_table=None):

    # Q-table 초기화
    if Q_table is None:
        Q = np.ones([env.observation_space.n, env.action_space.n])
    else:
        Q = Q_table

    # Episode별 reward 저장
    rList = []

    # Episode 반복
    for i in range(num_episodes):

        # 환경 초기화
        state = env.reset()[0]

        rAll = 0
        done = False

        # Step 반복
        for j in range(max_steps):

            # 현재 상태에서 최대 Q값 행동 선택
            action = np.argmax(Q[state,:])

            # 행동 수행
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # 오차 계산
            error = (
                reward
                + gamma * np.max(Q[next_state,:])
                - Q[state,action]
            )

            # Q 업데이트
            Q[state,action] = (
                Q[state,action]
                + learning_rate * error
            )

            rAll += reward
            state = next_state

            if done:
                break

        # Episode 총 reward 저장
        rList.append(rAll)

    return Q, rList


# 학습 실행
Q, rList = Q_learning(
    env,
    gamma,
    learning_rate,
    num_episodes,
    max_steps
)

# Reward 시각화
plt.plot(rList)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward')
plt.show()

# 후반부 reward 시각화
plt.plot(rList[2000:])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward After Episode 2000')
plt.show()


# 학습 결과 시뮬레이션
def simulate_Q(Q):

    test_seed = 42

    # Taxi-v4로 수정
    env_test = gym.make(
        'Taxi-v4',
        render_mode="rgb_array"
    )

    state = env_test.reset(seed=test_seed)[0]

    # 초기 화면 저장
    screen = env_test.render()
    screen_queue = [screen]

    # 최대 100 step
    for i in range(100):

        # 가장 좋은 행동 선택
        action = np.argmax(Q[state,:])

        # 행동 수행
        state, reward, terminated, truncated, info = env_test.step(action)

        done = terminated or truncated

        # 화면 저장
        screen = env_test.render()
        screen_queue.append(screen)

        if done:
            print(f"{i+1} timesteps 만에 종료되었습니다.")
            break

    env_test.close()

    return screen_queue


# 1 episode 결과 시각화
screen_queue_1ep = simulate_Q(Q_1ep)

fig = plt.figure()
ims = []

for i in range(len(screen_queue_1ep)):
    im = plt.imshow(screen_queue_1ep[i], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=300,
    blit=True,
    repeat_delay=1000
)

HTML(ani.to_jshtml())


# 최종 학습 결과 시각화
screen_queue = simulate_Q(Q)

fig = plt.figure()
ims = []

for i in range(len(screen_queue)):
    im = plt.imshow(screen_queue[i], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=300,
    blit=True,
    repeat_delay=4000
)

HTML(ani.to_jshtml())

# GIF 저장
ani.save('taxi-v4.gif', writer='Pillow', fps=2)

# Q-table CSV 저장
df = pd.DataFrame(Q)
df.to_csv('taxi-v4.csv', index=False, header=False)

print("학습 완료")