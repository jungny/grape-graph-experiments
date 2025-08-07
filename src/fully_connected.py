import random
import argparse
from itertools import combinations
from collections import deque
from tqdm import tqdm
import csv
import os
import time

def generate_preference_relation(degree, num_tasks):
    """
    SPAO 조건 만족하는 preference relation을 무작위로 생성.
    반환형: Tuple of Tuple[int, int] → ((task_id, num_neighbors), ...)
    """
    
    # 1. task별로 num_participants 후보 생성
    # degree가 주어졌을 때 num_participants ∈ {1, ..., degree+1}
    # 예: degree = 2, num_tasks = 3 이면
    # task_sorted = {
    #     1: [[1,1], [1,2], [1,3]],
    #     2: [[2,1], [2,2], [2,3]],
    #     3: [[3,1], [3,2], [3,3]],
    # }
    task_sorted = {
        task: [[task, num_participants] for num_participants in range(1, degree + 2)]
        for task in range(1, num_tasks + 1)
    }

    # 2. task 내에서 num_participants 오름차순 정렬 (SPAO 조건용)
    # 이건 위에서 이미 순서대로 만들었지만 혹시 모르니 정렬함
    for task in task_sorted:
        task_sorted[task].sort(key=lambda x: x[1])

    # 3. task별 preference들을 deque로 변환 (double-ended queue)
    # 예:
    # task_queues = {
    #     1: deque([[1,1], [1,2], [1,3]]),
    #     2: deque([[2,1], [2,2], [2,3]]),
    #     3: deque([[3,1], [3,2], [3,3]])
    # }
    task_queues = {task: deque(prefs) for task, prefs in task_sorted.items()}
    available_tasks = list(task_queues.keys())

    final_preferences = []

    # 4. 모든 큐가 빌 때까지 반복:
    # 남아 있는 task 중 랜덤으로 하나 골라서 맨 앞 요소 pop해서 리스트에 저장
    # → 이로 인해 task 간 순서는 랜덤, task 내 순서는 SPAO 보장
    while any(task_queues.values()):
        valid_tasks = [t for t in available_tasks if task_queues[t]]
        chosen_task = random.choice(valid_tasks)
        final_preferences.append(task_queues[chosen_task].popleft())

    # 예: 최종 결과
    # final_preferences = [
    #     [2, 1], [1, 1], [2, 2],
    #     [3, 1], [1, 2], [3, 2],
    #     [1, 3], [2, 3], [3, 3]
    # ]

    # 튜플로 변환해서 불변성 부여
    return tuple(tuple(pair) for pair in final_preferences)


def generate_random_scenario(seed):
    """
    - Seed로 랜덤 고정
    - Agent 수: 80, 160, 240, 320 중 하나
    - Task 수: 5, 10, 15, 20 중 하나
      (Task ID는 1부터 시작, 0은 void task → 제외)
    - Edge는 density 0.0 ~ 1.0 사이의 랜덤 값으로 설정
    - Agent 간의 Edge는 무작위로 선택
    - Agent마다 degree 기반 preference relation 생성
    """

    random.seed(seed)
    
    # agent_options = [5]
    # task_options = [3]
    agent_options = [80, 160, 240, 320]
    task_options = [5, 10, 15, 20]
    num_agents = random.choice(agent_options)
    num_tasks = random.choice(task_options)

    agents = list(range(num_agents))
    tasks = list(range(1, num_tasks+1)) # 0 is reserved for the void task

    # Generate all possible agent pairs
    all_pairs = list(combinations(agents, 2)) # (0, 1), (0, 2), ..., (num_agents-2, num_agents-1), always (a, b) where a < b
    total_possible = len(all_pairs)

    # Randomly select edges between agents
    density = random.uniform(0.0, 1.0)
    num_edges = int(total_possible * density)
    edges = set(random.sample(all_pairs, num_edges))

    # Calculate degrees of each agent
    degrees = {i: 0 for i in agents}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1

    # Create a preference relation for each agent
    preferences = {}
    for agent in agents:
        preferences[agent] = generate_preference_relation(degrees[agent], num_tasks)

    # Initialize allocation: all agents start with void task (0)
    allocation = {agent: 0 for agent in range(num_agents)}

    return {
        "num_agents": num_agents,
        "num_tasks": num_tasks,
        "allocation": allocation,
        "edges": edges,
        "preferences": preferences,
        "density": density, # for debugging
        "degrees": degrees  # for debugging
    }

def find_dissatisfied_agents(scenario):
    num_agents = scenario['num_agents']
    preferences = scenario['preferences']
    edges = scenario['edges']
    allocation = scenario['allocation']

    # agent 연결 정보 만들기 (edge 기반)
    connected = {agent: set() for agent in range(num_agents)}
    for i, j in edges:
        connected[i].add(j)
        connected[j].add(i)

    dissatisfied_agents = set()

    for agent_id in range(num_agents):
        current_task = allocation[agent_id]
        
        # 연결된 agent들의 task 할당 현황 조사
        task_counts = {}
        for other_id in connected[agent_id]:
            other_task = allocation[other_id]
            task_counts[other_task] = task_counts.get(other_task, 0) + 1

        # agent_id가 각 task로 이동한다고 가정하여 count + 1
        task_preferences = []
        for task_id in range(1, scenario['num_tasks'] + 1):
            num_participants = task_counts.get(task_id, 0)
            task_preferences.append((task_id, num_participants + 1))

        # preference relation에서 가장 왼쪽에 있는 pair 찾기
        agent_pref = preferences[agent_id]
        best_index = float('inf')
        best_task = None

        for task_pair in agent_pref:
            try:
                idx = agent_pref.index(task_pair)
                if idx < best_index:
                    best_index = idx
                    best_task = task_pair[0]
            except ValueError:
                print(f"Warning: Task pair {task_pair} not found in agent {agent_id}'s preferences")
                continue # task_pair가 agent_pref에 없을 경우

        if best_task is not None and best_task != current_task:
            dissatisfied_agents.add((agent_id, best_task))

    return dissatisfied_agents





def grape_allocation(scenario):
    allocation = scenario['allocation']
    num_agents = scenario['num_agents']

    iteration = 0
    threshold = num_agents ** 10  # 무한루프 방지용

    start_time = time.time()
    last_report_time = start_time

    while True:
        dissatisfied_agents = find_dissatisfied_agents(scenario)

        if not dissatisfied_agents:
            return {
                "allocation": allocation,
                "iteration": iteration,
                "NS": True
            }

        if iteration > threshold:
            raise RuntimeError(f"❌ Threshold {threshold} 초과: NS 도달 실패 (iteration: {iteration})")

        # 💡 특정 간격마다 진행상황 출력
        if iteration > 0 and iteration % num_agents == 0:
            now = time.time()
            elapsed = now - last_report_time
            total_elapsed = now - start_time
            tqdm.write(f"⏱️Iteration {iteration} (agent x {iteration // num_agents}) | "
                       f"+{elapsed:.2f}s since last | total {total_elapsed:.2f}s")
            last_report_time = now

        # 무작위 dissatisfied agent 선택 → best_task로 할당
        agent_id, best_task = random.choice(list(dissatisfied_agents))
        allocation[agent_id] = best_task
        iteration += 1


def main(start_seed, num_seeds=1000):
    print(f"Starting simulation for seeds {start_seed} to {start_seed + num_seeds - 1}")
    print("Each seed corresponds to a task allocation scenario.\n"
          "If everything works well, each seed should reach a 🌐 Nash Stable (NS) state.\n")

    csv_path = os.path.join("logs", "seed_info.csv")

    # tqdm 진행률 표시줄
    for i in tqdm(range(num_seeds),
                  desc=f"🌱 Seed {start_seed} ~ {start_seed + num_seeds - 1}",
                  unit="seed",
                  ncols=100,
                  colour='green'):

        seed = start_seed + i
        scenario = generate_random_scenario(seed)

        num_agents = scenario["num_agents"]
        num_tasks = scenario["num_tasks"]
        density = scenario["density"]
        num_edges = len(scenario["edges"])

        tqdm.write(f"🧪 Seed {seed}: {num_agents} agents, {num_tasks} tasks, "
           f"density {density * 100:.0f}%, {num_edges} edges")


        try:
            result = grape_allocation(scenario)
        except RuntimeError as e:
            tqdm.write(f"❌ Seed {seed} 실패: {str(e)}")
            continue

        allocation = result['allocation']
        iteration = result['iteration']
        NS = result['NS']

        if NS:
            tqdm.write(f"✅ Seed {seed}: Nash Equilibrium 도달 (iteration: {iteration})")

            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, num_agents, num_tasks, density, num_edges, iteration])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, help="설명")
    parser.add_argument("--num_seeds", type=int, default=1000, help="Number of iterations to run")
    args = parser.parse_args()
    main(args.start_seed, args.num_seeds)