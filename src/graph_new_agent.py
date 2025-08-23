import random
import argparse
from itertools import combinations
from collections import deque
from tqdm import tqdm
import csv
import os
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v2 as imageio

LOG_PATH = os.path.join("logs", "run_log.txt")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log(message, log_path=None):
    tqdm.write(message)  # 터미널에 출력
    if log_path is None:
        log_path = LOG_PATH
    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write(message + '\n')  # 파일에도 저장

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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
    - Agent 수: 5~160 사이 랜덤
    - Task 수: 5~10 사이 랜덤
    - Edge는 density 0%~100% 사이 랜덤
    - Agent 간의 Edge는 무작위로 선택
    - Agent마다 degree 기반 preference relation 생성
    """

    random.seed(seed)
    
    agent_options = list(range(5, 161))  # 5~160
    task_options = list(range(5, 11))   # 5~10
    num_agents = random.choice(agent_options)
    num_tasks = random.choice(task_options)

    agents = list(range(num_agents))
    tasks = list(range(1, num_tasks+1)) # 0 is reserved for the void task

    # Generate all possible agent pairs
    all_pairs = list(combinations(agents, 2)) # (0, 1), (0, 2), ..., (num_agents-2, num_agents-1), always (a, b) where a < b
    total_possible = len(all_pairs)

    # Density를 0%~100% 사이 랜덤으로 설정
    density = random.uniform(0.0, 1.0)
    num_edges = int(total_possible * density)
    edges = set(random.sample(all_pairs, num_edges))  # 랜덤하게 edge 선택

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

    # 연결 정보 생성 (edge 기반)
    connected = {agent: set() for agent in agents}
    for i, j in edges:
        connected[i].add(j)
        connected[j].add(i)

    return {
        "num_agents": num_agents,
        "num_tasks": num_tasks,
        "allocation": allocation,
        "edges": edges,
        "preferences": preferences,
        "connected": connected,
        "density": density, # for debugging
        "degrees": degrees  # for debugging
    }

def find_dissatisfied_agents(scenario, last_moved_agent=None, last_from_task=None, last_to_task=None, cause_map=None):
    num_agents  = scenario['num_agents']
    preferences = scenario['preferences']
    allocation  = scenario['allocation']
    connected   = scenario['connected']

    dissatisfied_agents = set()
    for agent_id in range(num_agents):
        current_task = allocation[agent_id]
        agent_pref   = preferences[agent_id]

        task_counts = {}
        for other_id in connected[agent_id]:
            other_task = allocation[other_id]
            task_counts[other_task] = task_counts.get(other_task, 0) + 1

        current_key = (current_task, task_counts.get(current_task, 0) + 1)
        try:
            current_rank = agent_pref.index(current_key)
        except ValueError:
            current_rank = float('inf')

        best_task = None
        best_rank = current_rank

        for task_id in range(1, scenario['num_tasks'] + 1):
            key = (task_id, task_counts.get(task_id, 0) + 1)
            try:
                new_rank = agent_pref.index(key)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_task = task_id
            except ValueError:
                continue

        if best_task is not None:
            dissatisfied_agents.add((agent_id, best_task))
            
            # Track cause agents and move types if cause_map is provided
            if cause_map is not None and last_moved_agent is not None:
                # If the agent's current task == last_to_task: This agent is being pushed by last_moved_agent
                if current_task == last_to_task:
                    cause_map[agent_id] = (last_moved_agent, "pushed")
                # If the agent's best_task == last_from_task: This agent is being pulled by last_moved_agent
                elif best_task == last_from_task:
                    cause_map[agent_id] = (last_moved_agent, "pulled")

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

        # num_agents의 정수배마다 진행상황 출력
        if iteration > 0 and iteration % num_agents == 0:
            now = time.time()
            elapsed = now - last_report_time
            total_elapsed = now - start_time
            dissatisfied_ratio = len(dissatisfied_agents) / num_agents * 100

            log(
                f"🐜 Iteration {iteration} "
                f"(num_agents x {iteration // num_agents}) | "
                f"dissatisfied: {dissatisfied_ratio:.1f}% | "
                f"From last log: {elapsed:.2f}s | Total: {total_elapsed:.2f}s"
            )
            last_report_time = now

        # 무작위 dissatisfied agent 선택 → best_task로 할당
        agent_id, best_task = random.choice(list(dissatisfied_agents))
        allocation[agent_id] = best_task
        iteration += 1

def grape_allocation_with_history(scenario, sample_every=1, report_every=100, log_path=None):
    allocation = scenario['allocation']
    num_agents = scenario['num_agents']
    iteration = 0
    threshold = num_agents ** 10

    history = []                  # allocations snapshot
    highlights = []               # each snapshot's "last changed agent"
    history.append([allocation[i] for i in range(num_agents)])
    highlights.append(None)       # 초기 프레임은 하이라이트 없음

    last_report_time = time.time()
    last_changed_agent = None

    while True:
        dissatisfied_agents = find_dissatisfied_agents(scenario)

        if not dissatisfied_agents:
            log(f"[GRAPE] NS reached | iter={iteration}", log_path)
            return {
                "allocation": allocation,
                "iteration": iteration,
                "NS": True,
                "history": history,
                "highlights": highlights
            }

        if iteration > threshold:
            raise RuntimeError(f"Threshold {threshold} 초과: NS 도달 실패 (iteration: {iteration})")

        agent_id, best_task = random.choice(list(dissatisfied_agents))
        allocation[agent_id] = best_task
        last_changed_agent = agent_id            # <- 방금 바꾼 에이전트
        iteration += 1

        if iteration % sample_every == 0:
            history.append([allocation[i] for i in range(num_agents)])
            highlights.append(last_changed_agent)

        if iteration % report_every == 0:
            now = time.time()
            dissatisfied_ratio = len(dissatisfied_agents) / num_agents * 100
            log(f"[GRAPE] iter={iteration} | dissatisfied~={dissatisfied_ratio:.1f}% | +{now-last_report_time:.2f}s", log_path)
            last_report_time = now

def grape_allocation_with_moves(scenario, seed, sample_every=1, report_every=100, log_path=None):
    allocation = scenario['allocation']
    num_agents = scenario['num_agents']
    iteration = 0
    threshold = num_agents ** 10

    moves = []                    # list of move tuples
    total_moves = 0
    pushed_moves = 0
    pulled_moves = 0

    # Initialize cause tracking
    scenario['cause_map'] = {}
    last_moved_agent = None
    last_from_task = None
    last_to_task = None

    last_report_time = time.time()

    while True:
        dissatisfied_agents = find_dissatisfied_agents(
            scenario, 
            last_moved_agent, 
            last_from_task, 
            last_to_task, 
            scenario['cause_map']
        )

        if not dissatisfied_agents:
            log(f"[GRAPE] NS reached | iter={iteration}", log_path)
            
            # Check if this is a rare case
            num_agents_after = scenario['num_agents']
            is_rare_case = (pulled_moves >= 2 and total_moves >= (num_agents_after / 2))
            
            # Always update summary CSV
            summary_csv_path = os.path.join("logs", "summary.csv")
            ensure_dir(os.path.dirname(summary_csv_path))
            
            # Get number of agents before adding new agent
            num_agents_before = num_agents_after - 1
            
            # Write header if file doesn't exist
            write_header = not os.path.exists(summary_csv_path) or os.path.getsize(summary_csv_path) == 0
            with open(summary_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(["seed", "num_agents_before", "num_tasks", "total_moves", "pushed_moves", "pulled_moves"])
                writer.writerow([seed, num_agents_before, scenario['num_tasks'], total_moves, pushed_moves, pulled_moves])
            
            # If rare case, save detailed move files
            if is_rare_case:
                detailed_dir = os.path.join("logs", "graph_new_agent", "moves_detailed")
                ensure_dir(detailed_dir)
                
                # Save detailed CSV
                csv_path = os.path.join(detailed_dir, f"seed_{seed}.csv")
                with open(csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["moved_agent", "move_type", "from_task", "to_task", "cause_agent"])
                    for move in moves:
                        writer.writerow(move)
                
                # Save human-readable text
                txt_path = os.path.join(detailed_dir, f"seed_{seed}.txt")
                with open(txt_path, mode='w', encoding='utf-8') as file:
                    for move in moves:
                        moved_agent, move_type, from_task, to_task, cause_agent = move
                        if move_type == "pushed":
                            file.write(" " * 40 + f"a{moved_agent} pushed ({from_task}→{to_task}) by a{cause_agent}\n")
                        else:  # pulled
                            file.write(f"a{moved_agent} pulled ({from_task}→{to_task}) by a{cause_agent}\n")
                
                log(f"[GRAPE] Rare case detected! Detailed moves saved to {detailed_dir}", log_path)
            
            return {
                "allocation": allocation,
                "iteration": iteration,
                "NS": True,
                "moves": moves,
                "total_moves": total_moves,
                "pushed_moves": pushed_moves,
                "pulled_moves": pulled_moves
            }

        if iteration > threshold:
            raise RuntimeError(f"Threshold {threshold} 초과: NS 도달 실패 (iteration: {iteration})")

        agent_id, best_task = random.choice(list(dissatisfied_agents))
        from_task = allocation[agent_id]
        allocation[agent_id] = best_task
        
        # Retrieve cause agent and move type from cause_map
        cause_info = scenario['cause_map'].get(agent_id, (None, None))
        cause_agent, move_type = cause_info
        
        # If move_type is not available from cause_map, fall back to basic classification
        if move_type is None:
            if from_task == 0:  # Agent was unassigned
                move_type = "pulled"
            else:  # Agent was already assigned to a task
                move_type = "pushed"
        
        # Record the move
        move_tuple = (agent_id, move_type, from_task, best_task, cause_agent)
        moves.append(move_tuple)
        total_moves += 1
        
        if move_type == "pushed":
            pushed_moves += 1
        else:  # pulled
            pulled_moves += 1
        
        # Update tracking variables for next iteration
        last_moved_agent = agent_id
        last_from_task = from_task
        last_to_task = best_task
        
        iteration += 1

        if iteration % report_every == 0:
            now = time.time()
            dissatisfied_ratio = len(dissatisfied_agents) / num_agents * 100
            log(f"[GRAPE] iter={iteration} | dissatisfied~={dissatisfied_ratio:.1f}% | "
                f"moves: {total_moves} (pushed: {pushed_moves}, pulled: {pulled_moves}) | "
                f"+{now-last_report_time:.2f}s", log_path)
            last_report_time = now

# ======================================
# 시각화 관련 함수들 (fc_with_vis.py에서 가져옴)
# ======================================
def spring_layout_from_edges(num_agents, edges, k=None, iterations=300, spread=1.6, seed=42):
    """
    간단한 Fruchterman-Reingold force-directed 레이아웃 (의존성 없음).
    - k: 이상적인 간선 길이 (None이면 sqrt(area/n))
    - iterations: 반복 횟수 (늘릴수록 더 퍼짐/안정)
    - spread: 최종 스케일 배수 (값 키우면 더 퍼짐)
    """
    rng = np.random.default_rng(seed)
    # 초기 위치: 작은 난수 구름
    pos = rng.normal(scale=1e-3, size=(num_agents, 2)).astype(float)

    # 인접행렬 (bool)
    A = np.zeros((num_agents, num_agents), dtype=bool)
    for a, b in edges:
        A[a, b] = True; A[b, a] = True

    # 파라미터 세팅
    area = 1.0
    if k is None:
        k = np.sqrt(area / max(1, num_agents))  # 이론적 적정 거리
    t = 0.1  # 초기 "온도"
    cool = 0.95  # 냉각율

    for _ in range(iterations):
        # 모든 쌍에 대한 벡터 (N,N,2) — N<=320이면 충분히 감당 가능
        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = (delta**2).sum(axis=2) + 1e-9
        dist = np.sqrt(dist2)

        # 반발력 (모든 쌍)
        rep = (k*k / dist2)[:, :, None] * delta

        # 인력 (간선만)
        att = np.zeros_like(rep)
        if A.any():
            att[A] = (dist[A][:, None]**2 / k) * (delta[A] / dist[A][:, None])

        disp = (rep - att).sum(axis=1)

        # 위치 업데이트 + 온도 제한
        disp_norm = np.linalg.norm(disp, axis=1) + 1e-12
        pos += (disp / disp_norm[:, None]) * np.minimum(disp_norm, t)[:, None]
        # 중심화
        pos -= pos.mean(axis=0, keepdims=True)
        # 냉각
        t *= cool

    # 스케일 업 (퍼짐 정도)
    pos = pos / (np.abs(pos).max() + 1e-12) * (300 * spread)
    return pos

def build_environment_for_vis(num_agents, num_tasks, edges, layout="spring", spread=1.6):
    if layout == "spring":
        agent_locations = spring_layout_from_edges(num_agents, edges, iterations=300, spread=spread, seed=42)
    else:
        # spectral layout는 생략하고 spring만 사용
        agent_locations = spring_layout_from_edges(num_agents, edges, iterations=300, spread=spread, seed=42)

    rng = np.random.default_rng(42)
    agent_resources = rng.uniform(10, 60, size=num_agents)

    agent_comm_matrix = np.zeros((num_agents, num_agents), dtype=int)
    for a, b in edges:
        agent_comm_matrix[a, b] = 1
        agent_comm_matrix[b, a] = 1
    np.fill_diagonal(agent_comm_matrix, 0)

    environment = {
        'agent_locations': agent_locations,
        'agent_resources': agent_resources,
        'num_tasks': num_tasks
    }
    return environment, agent_comm_matrix

def visualise_agents_only(
    scenario,
    final_allocation=None,
    filename="result_vis.png",
    edge_mode='none',
    edge_max=2000,
    edge_alpha=0.06,
    agent_size=4,
    highlight_agent=None,
    highlight_edge_lw=1.5,
    highlight_node_ms=8
):
    env = scenario['environment']
    agent_locations  = env['agent_locations']
    num_tasks        = env['num_tasks']
    agent_comm_matrix= scenario['agent_comm_matrix']

    # 색(할당 없음=검정)
    colours = plt.cm.gist_rainbow(np.linspace(0, 1, num_tasks))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True); ax.set_aspect('equal')
    # 범례가 그래프 오른쪽 바깥에 위치할 수 있도록 우측 여백 확보
    plt.subplots_adjust(right=0.78)
    m = np.max(np.abs(agent_locations)) if len(agent_locations)>0 else 1.0
    ax.set_xlim([-m*1.2, m*1.2]); ax.set_ylim([-m*1.2, m*1.2]); ax.set_xlabel('X'); ax.set_ylabel('Y')

    # 일반 엣지 (샘플이지만 사실상 all)
    if edge_mode == 'sample':
        iu = np.triu_indices(agent_comm_matrix.shape[0], k=1)
        pairs = [(i,j) for i,j in zip(iu[0], iu[1]) if agent_comm_matrix[i,j] > 0]
        for i, j in pairs:
            ax.plot([agent_locations[i,0], agent_locations[j,0]],
                    [agent_locations[i,1], agent_locations[j,1]],
                    '-', color='#454545', linewidth=0.3, alpha=edge_alpha, zorder = 1)

    # 에이전트 점
    for i in range(agent_locations.shape[0]):
        alloc = final_allocation[i] if final_allocation is not None else 0
        ms = agent_size if i != highlight_agent else highlight_node_ms
        if alloc == 0:
            facecolor = 'white'    # 내부 흰색
            edgecolor = 'black'    # 검은 테두리
            ax.plot(agent_locations[i,0], agent_locations[i,1], 'o',
                    markersize=ms, markeredgewidth=0.8,
                    markeredgecolor=edgecolor, markerfacecolor=facecolor, zorder=3)
        else:
            ax.plot(agent_locations[i,0], agent_locations[i,1], 'o',
                    markersize=ms, markeredgewidth=0,   # 테두리 없음
                    markerfacecolor=colours[int(alloc-1)], zorder=3)

    # 하이라이트: 선택된 에이전트의 모든 인접 엣지를 검정/굵게
    if highlight_agent is not None:
        nbrs = np.where(agent_comm_matrix[highlight_agent] > 0)[0]
        for j in nbrs:
            ax.plot([agent_locations[highlight_agent,0], agent_locations[j,0]],
                    [agent_locations[highlight_agent,1], agent_locations[j,1]],
                    '-', color='#2E2E2E', linewidth=highlight_edge_lw, alpha=0.9, zorder=2)

    # 범례: 색 → Task 매핑 + 할당 없음
    legend_elements = []
    # Unassigned (void)
    legend_elements.append(
        Line2D([0], [0], marker='o', color='none', label='Unassigned',
               markerfacecolor='white', markeredgecolor='black', markersize=6)
    )
    # Tasks 1..num_tasks
    for t in range(1, num_tasks + 1):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='none', label=f'Task {t}',
                   markerfacecolor=colours[t-1], markeredgecolor='none', markersize=6)
        )
    ax.legend(
        handles=legend_elements,
        title='Legend',
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True
    )

    title_name = 'Task Allocation Result' if final_allocation is not None else 'Initial Layout'
    plt.title(title_name)
    if filename:
        plt.savefig(filename, dpi=160)
        plt.close()

def generate_gif_from_history(
    scenario,
    allocation_history,
    filename='result_animation.gif',
    step=1,
    duration=0.3,
    log_path=None,
    highlights=None,
    edge_mode='sample',
    edge_max=2000,
    edge_alpha=0.12,
    highlight_edge_lw=0.5,
    agent_size=4
):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(filename), "_tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    if highlights is None:
        highlights = [None]*len(allocation_history)

    log(f"[GIF] writing frames to {tmp_dir}", log_path)
    images = []
    idxs = list(range(0, len(allocation_history), step))
    for k in tqdm(idxs, desc="[GIF] frames"):
        frame_path = os.path.join(tmp_dir, f"frame_{k:06d}.png")
        visualise_agents_only(
            scenario,
            allocation_history[k],
            filename=frame_path,
            highlight_agent=highlights[k],
            edge_mode=edge_mode,
            edge_max=edge_max,
            edge_alpha=edge_alpha,
            highlight_edge_lw=highlight_edge_lw,
            agent_size=agent_size
        )
        images.append(frame_path)

    log(f"[GIF] assembling {len(images)} frames → {filename}", log_path)
    with imageio.get_writer(filename, mode='I', duration=duration) as writer:
        for p in tqdm(images, desc="[GIF] assembling"):
            writer.append_data(imageio.imread(p))

    log("[GIF] done", log_path)

def run_experiment(seed, visualize=False):
    """
    메인 실험 함수:
    1. 초기 Nash equilibrium까지 GRAPE 실행
    2. 랜덤 에이전트 선택하여 preference relation 초기화
    3. 새로운 preference로 다시 GRAPE 실행하여 iteration 수 기록
    """
    
    if visualize:
        # 시각화 모드: logs/newpr/seed 폴더에 결과 저장
        save_dir = os.path.join("logs", "newpr", str(seed))
        ensure_dir(save_dir)
        log_path = os.path.join(save_dir, "run_log_pr.txt")
    else:
        # CSV 기록 모드: logs 폴더에 로그만 저장
        log_path = LOG_PATH

    log(f"[EXPERIMENT] seed={seed}", log_path)
    
    # 1. 초기 시나리오 생성
    log("[EXPERIMENT] generating initial scenario...", log_path)
    scenario = generate_random_scenario(seed)
    log(f"[EXPERIMENT] agents={scenario['num_agents']}, tasks={scenario['num_tasks']}, "
        f"density={scenario['density']:.3f}, edges={len(scenario['edges'])}", log_path)

    # 2. 초기 Nash equilibrium까지 GRAPE 실행
    log("[EXPERIMENT] running initial GRAPE to reach Nash equilibrium...", log_path)
    initial_result = grape_allocation(scenario)
    log(f"[EXPERIMENT] Initial Nash equilibrium reached! Iterations: {initial_result['iteration']}", log_path)

    # 3. 새로운 에이전트 추가
    num_agents_before = scenario['num_agents']
    new_agent = num_agents_before
    scenario['num_agents'] += 1
    
    # 새로운 에이전트를 위한 연결 정보 초기화
    scenario['connected'][new_agent] = set()
    
    # 새로운 에이전트의 연결을 위한 새로운 density 생성 (0% 초과 100% 미만)
    new_density = random.uniform(0.0, 1.0)
    
    # 새로운 에이전트와 기존 에이전트들 간의 edge 생성
    for existing_agent in range(num_agents_before):
        if random.random() < new_density:  # 새로운 density 확률로 연결
            scenario['connected'][new_agent].add(existing_agent)
            scenario['connected'][existing_agent].add(new_agent)
            # edge 순서를 (작은 번호, 큰 번호)로 정렬하여 추가
            edge_pair = (min(new_agent, existing_agent), max(new_agent, existing_agent))
            scenario['edges'].add(edge_pair)
    
    # 새로운 에이전트의 실제 degree 계산
    actual_degree = len(scenario['connected'][new_agent])
    
    # 새로운 에이전트의 preference relation 생성 (실제 degree 기반)
    new_pref = generate_preference_relation(actual_degree, scenario['num_tasks'])
    scenario['preferences'][new_agent] = new_pref
    
    # 새로운 에이전트의 초기 할당을 preference에서 가장 높은 순위의 task로 설정
    best_initial_task = new_pref[0][0]
    scenario['allocation'][new_agent] = best_initial_task
    
    log(f"[EXPERIMENT] New agent a{new_agent} added → init task {best_initial_task} | "
        f"new_density={new_density:.3f} | actual_degree={actual_degree}", log_path)

    # 4. 시각화 모드인 경우 환경 설정
    if visualize:
        log(f"[EXPERIMENT] setting up visualization environment...", log_path)
        env, A = build_environment_for_vis(
            scenario['num_agents'],
            scenario['num_tasks'],
            scenario['edges'],
            layout="spring",
            spread=1.8
        )
        scenario_vis = {
            'environment': env,
            'agent_comm_matrix': A,
            'num_agents': scenario['num_agents'],
            'num_tasks': scenario['num_tasks']
        }
        
        # 초기 Nash equilibrium 상태 시각화
        initial_alloc = [initial_result['allocation'][i] for i in range(scenario['num_agents'])]
        visualise_agents_only(
            scenario_vis, 
            initial_alloc, 
            filename=os.path.join(save_dir, "fig_initial_nash.png"),
            edge_mode='sample'
        )

    # 5. 새로운 preference로 다시 GRAPE 실행 (히스토리 포함)
    log("[EXPERIMENT] running GRAPE with new preference to count iterations...", log_path)
    new_result = grape_allocation_with_history(
        scenario, 
        sample_every=1, 
        report_every=max(100, scenario['num_agents']),
        log_path=log_path
    )
    
    new_iterations = new_result['iteration']
    log(f"[EXPERIMENT] New Nash equilibrium reached! Iterations: {new_iterations}", log_path)

    # 6. 시각화 모드인 경우 결과 시각화 및 GIF 생성
    if visualize:
        # 최종 상태 시각화
        final_alloc = [new_result['allocation'][i] for i in range(scenario['num_agents'])]
        visualise_agents_only(
            scenario_vis, 
            final_alloc, 
            filename=os.path.join(save_dir, "fig_final_nash.png"),
            edge_mode='sample'
        )

        # GIF 생성
        gif_path = os.path.join(save_dir, f"preference_change_animation_{seed}.gif")
        generate_gif_from_history(
            scenario_vis,
            new_result['history'],
            filename=gif_path,
            step=1,
            duration=0.25,
            log_path=log_path,
            highlights=new_result.get('highlights'),
            edge_mode='sample',
            edge_alpha=0.06,
            highlight_edge_lw=0.5
        )
        
        log(f"[EXPERIMENT] Visualization saved to {save_dir}", log_path)
        return {
            "seed": seed,
            "num_agents": scenario['num_agents'],
            "num_tasks": scenario['num_tasks'],
            "changed_agent": random_agent,
            "initial_iterations": initial_result['iteration'],
            "new_iterations": new_iterations,
            "visualization_dir": save_dir
        }
    else:
        # CSV 기록 모드
        return {
            "seed": seed,
            "num_agents": scenario['num_agents'],
            "num_tasks": scenario['num_tasks'],
            "density": scenario['density'],
            "changed_agent": random_agent,
            "initial_iterations": initial_result['iteration'],
            "new_iterations": new_iterations
        }

def write_result_row(csv_path, row):
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["seed", "num_agents", "num_tasks", "density", "changed_agent", "initial_iterations", "new_iterations"])
        writer.writerow(row)

def main(start_seed, num_seeds=1000, visualize=False):
    print(f"Starting preference change experiment for seeds {start_seed} to {start_seed + num_seeds - 1}")
    print("Each seed corresponds to a task allocation scenario.\n"
          "1. Generate initial Nash equilibrium\n"
          "2. Change one agent's preference randomly\n"
          "3. Count iterations to reach new Nash equilibrium\n")

    if visualize:
        print("🔍 Visualization mode: Results will be saved to logs/newpr/ folders")
        csv_path = None
    else:
        print("📊 CSV mode: Results will be saved to logs/seed_info.csv")
        csv_path = os.path.join("logs/csv", "seed_info_pr.csv")
        ensure_dir(os.path.dirname(csv_path))

    # tqdm 진행률 표시줄
    for i in tqdm(range(num_seeds),
                  desc=f"🐫 Seed {start_seed} ~ {start_seed + num_seeds - 1}",
                  unit="seed",
                  ncols=100,
                  colour='green',
                  smoothing=0.05
                  ):

        seed = start_seed + i
        
        try:
            result = run_experiment(seed, visualize=visualize)
        except Exception as e:
            log(f"❌ Seed {seed} 실패: {str(e)}")
            continue

        if not visualize and csv_path:
            # CSV에 결과 기록
            write_result_row(csv_path, [
                result['seed'], 
                result['num_agents'], 
                result['num_tasks'], 
                result['density'],
                result['changed_agent'],
                result['initial_iterations'],
                result['new_iterations']
            ])
        
        if i % 100 == 0:
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, required=True, help="시작 시드 번호")
    parser.add_argument("--num_seeds", type=int, default=1000, help="실행할 시드 개수")
    parser.add_argument("--visualize", action="store_true", help="시각화 모드 활성화 (CSV 기록 안함)")
    args = parser.parse_args()
    
    main(args.start_seed, args.num_seeds, args.visualize)