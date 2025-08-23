# fc_with_vis.py
import random
import argparse
from itertools import combinations
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v2 as imageio
import csv
import os
import time
from tqdm import tqdm

# ===============================
# 0) 공통 유틸
# ===============================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log(msg, log_path=None):
    tqdm.write(msg)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ===============================
# 1) fully_connected 파트 (원형)
# ===============================
def generate_preference_relation(degree, num_tasks):
    task_sorted = {
        task: [[task, num_participants] for num_participants in range(1, degree + 2)]
        for task in range(1, num_tasks + 1)
    }
    for task in task_sorted:
        task_sorted[task].sort(key=lambda x: x[1])
    task_queues = {task: deque(prefs) for task, prefs in task_sorted.items()}
    available_tasks = list(task_queues.keys())

    final_preferences = []
    while any(task_queues.values()):
        valid_tasks = [t for t in available_tasks if task_queues[t]]
        chosen_task = random.choice(valid_tasks)
        final_preferences.append(task_queues[chosen_task].popleft())

    return tuple(tuple(pair) for pair in final_preferences)

def save_preferences_csv(preferences, save_dir, seed, num_tasks, degrees):
    """
    preferences: dict[int -> tuple[(task_id, num_participants), ...]]
    degrees: dict[int -> int]
    """
    # path = os.path.join(save_dir, f"preferences_{seed}.csv")
    # os.makedirs(save_dir, exist_ok=True)
    # with open(path, "w", newline="", encoding="utf-8") as f:
    #     w = csv.writer(f)
    #     # 컬럼 정의
    #     w.writerow(["seed", "agent_id", "rank", "task_id", "num_participants", "agent_degree", "num_tasks"])
    #     # 각 에이전트의 선호 순서대로 덤프
    #     for agent_id, pref_list in preferences.items():
    #         deg = degrees.get(agent_id, 0)
    #         for rank, (task_id, n_part) in enumerate(pref_list, start=1):
    #             w.writerow([seed, agent_id, rank, task_id, n_part, deg, num_tasks])
    pass
    # return path

def generate_random_scenario(seed):
    random.seed(seed)
    #agent_options = [16]
    agent_options = [80, 160]   # 240, 320 제외 가능
    task_options  = [5, 10, 15, 20]
    num_agents = random.choice(agent_options)
    num_tasks  = random.choice(task_options)

    agents = list(range(num_agents))
    all_pairs = list(combinations(agents, 2))
    total_possible = len(all_pairs)

    density = random.uniform(0.0, 1.0)
    num_edges = int(total_possible * density)
    edges = set(random.sample(all_pairs, num_edges))

    degrees = {i: 0 for i in agents}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1

    preferences = {agent: generate_preference_relation(degrees[agent], num_tasks) for agent in agents}
    allocation = {agent: 0 for agent in range(num_agents)}  # 0 = void
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
        "density": density,
        "degrees": degrees
    }

def find_dissatisfied_agents(scenario):
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


    return dissatisfied_agents

def grape_allocation_with_history(scenario, sample_every=10, report_every=1000, log_path=None):
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


# ======================================
# 2) 좌표/환경 생성 + 시각화(ORIGINAL식)
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


def spectral_layout_from_edges(num_agents, edges, eps=1e-9):
    A = np.zeros((num_agents, num_agents), dtype=float)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    idx_x = 1 if len(order) > 1 else 0
    idx_y = 2 if len(order) > 2 else idx_x
    X = evecs[:, order[idx_x]]
    Y = evecs[:, order[idx_y]]
    coords = np.stack([X, Y], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    maxabs = np.abs(coords).max() + eps
    coords = coords / maxabs * 300
    return coords

def build_environment_for_vis(num_agents, num_tasks, edges, layout="spring", spread=1.6):
    if layout == "spring":
        agent_locations = spring_layout_from_edges(num_agents, edges, iterations=300, spread=spread, seed=42)
    else:
        agent_locations = spectral_layout_from_edges(num_agents, edges)  # 백업용

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
    highlight_agent=None,            # <- 추가
    highlight_edge_lw=1.5,           # <- 추가
    highlight_node_ms=8              # <- 추가
):
    env = scenario['environment']
    agent_locations  = env['agent_locations']
    num_tasks        = env['num_tasks']
    agent_comm_matrix= scenario['agent_comm_matrix']

    # 색(할당 없음=검정)
    # colours = plt.cm.viridis(np.linspace(0, 1, num_tasks))
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
        # if len(pairs) > edge_max:
        #     pairs = random.sample(pairs, edge_max)
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
    edge_mode='sample',          # ← 추가
    edge_max=2000,               # ← 추가
    edge_alpha=0.12,             # ← 추가
    highlight_edge_lw=0.5,       # ← 추가
    agent_size=4                 # ← 추가
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
            edge_mode=edge_mode,           # ← 추가
            edge_max=edge_max,             # ← 추가
            edge_alpha=edge_alpha,         # ← 추가
            highlight_edge_lw=highlight_edge_lw,  # ← 추가
            agent_size=agent_size          # ← 추가
        )
        images.append(frame_path)

    log(f"[GIF] assembling {len(images)} frames → {filename}", log_path)
    with imageio.get_writer(filename, mode='I', duration=duration) as writer:
        for p in tqdm(images, desc="[GIF] assembling"):
            writer.append_data(imageio.imread(p))

    # for p in images:
    #     try: os.remove(p)
    #     except: pass
    # try: os.rmdir(tmp_dir)
    # except: pass
    log("[GIF] done", log_path)


# ===============================
# 3) 실행 래퍼
# ===============================
def run_once(seed, sample_every=10, gif_step=1, gif_path=None, edge_mode='none',
             layout="spring", spread=1.8):
    save_dir = os.path.join("logs", str(seed))
    ensure_dir(save_dir)
    log_path = os.path.join(save_dir, "run_log.txt")

    log(f"[RUN] seed={seed}", log_path)
    log("[RUN] generating scenario...", log_path)
    scenario_fc = generate_random_scenario(seed)
    log(f"[RUN] agents={scenario_fc['num_agents']}, tasks={scenario_fc['num_tasks']}, "
        f"density={scenario_fc['density']:.3f}, edges={len(scenario_fc['edges'])}", log_path)
    
    prefs_csv = save_preferences_csv(
        preferences=scenario_fc["preferences"],
        save_dir=save_dir,
        seed=seed,
        num_tasks=scenario_fc["num_tasks"],
        degrees=scenario_fc["degrees"]
    )
    log(f"[RUN] preferences saved -> {prefs_csv}", log_path)

    log(f"[RUN] computing layout ({layout}, spread={spread}) ...", log_path)
    env, A = build_environment_for_vis(
        scenario_fc['num_agents'],
        scenario_fc['num_tasks'],
        scenario_fc['edges'],
        layout=layout,
        spread=spread
    )
    scenario_vis = {
        'environment': env,
        'agent_comm_matrix': A,
        'num_agents': scenario_fc['num_agents'],
        'num_tasks': scenario_fc['num_tasks']
    }

    # 초기 상태 스냅샷
    visualise_agents_only(scenario_vis, None, filename=os.path.join(save_dir, "fig_initial.png"),
                          edge_mode=edge_mode)

    # fully_connected 알고리즘 실행 + 히스토리 기록
    log("[RUN] running GRAPE (fully_connected) ...", log_path)
    result = grape_allocation_with_history(
        scenario_fc, sample_every=sample_every, report_every=max(1000, scenario_fc['num_agents']),
        log_path=log_path
    )
    final_alloc = [result['allocation'][i] for i in range(scenario_fc['num_agents'])]
    log(f"[RUN] finished | iter={result['iteration']} | NS={result['NS']}", log_path)

    # 최종 스냅샷
    visualise_agents_only(scenario_vis, final_alloc, filename=os.path.join(save_dir, "fig_final.png"),
                          edge_mode=edge_mode)

    # GIF 생성
    if gif_path is None:
        gif_path = os.path.join(save_dir, f"result_animation_{seed}.gif")
    generate_gif_from_history(
        scenario_vis,
        result['history'],
        filename=gif_path,
        step=gif_step,
        duration=0.25,
        log_path=log_path,
        highlights=result.get('highlights'),
        edge_mode=edge_mode,           # ← 추가
        edge_alpha=0.06,               # 원하면 조절
        highlight_edge_lw=0.5         # 하이라이트 굵기 조절 포인트
    )

    return {
        "iteration": result["iteration"],
        "NS": result["NS"],
        "num_agents": scenario_fc["num_agents"],
        "num_tasks": scenario_fc["num_tasks"],
        "density": scenario_fc["density"],
        "gif": gif_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample_every", type=int, default=10, help="히스토리 샘플링 간격")
    parser.add_argument("--gif_step", type=int, default=1, help="GIF 프레임 간격(히스토리 인덱스 기준)")
    parser.add_argument("--gif_path", type=str, default=None)
    parser.add_argument("--edge_mode", type=str, default="sample", choices=["none", "sample"],
                        help="시각화에서 에이전트 간 엣지 표시 방식")
    parser.add_argument("--layout", type=str, default="spring", choices=["spring", "spectral"])
    parser.add_argument("--spread", type=float, default=1.8, help="레이아웃 퍼짐 정도 배수")
    args = parser.parse_args()

    info = run_once(
        args.seed,
        sample_every=args.sample_every,
        gif_step=args.gif_step,
        gif_path=args.gif_path,
        edge_mode=args.edge_mode,
        layout=args.layout,      # ← 추가
        spread=args.spread       # ← 추가
    )
    print(f"[done] seed={args.seed}, NS={info['NS']}, iter={info['iteration']}, "
          f"agents={info['num_agents']}, tasks={info['num_tasks']}, "
          f"density={info['density']:.3f}, gif={info['gif']}")
