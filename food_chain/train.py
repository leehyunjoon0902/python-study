"""학습: 초식(토끼 · 사슴) = SARSA, 육식(여우 · 늑대) = Q-Learning (종별 공유 Q-Table 4개).

다중 에이전트 비정상성을 줄이기 위한 교대 학습(schedule='alternating'):
    ① 초식 2종 학습 (육식은 규칙 기반 추적)        — 전체의 40%
    ② 육식 2종 학습 (초식 정책 고정)               — 전체의 40%
    ③ 10 episode씩 초식 ↔ 육식 번갈아 미세 조정 (ε = 최솟값) — 전체의 20%
schedule='joint'는 처음부터 모든 종이 동시에 학습한다(비교용).

train 모드 세계: 번식 OFF, 죽으면 같은 종을 무작위 위치에 즉시 재생성,
episode마다 초기 개체수 · 풀 재생 속도를 무작위화한다. episode 길이 제한은 terminal이 아니다.
개체수는 config.TRAIN_COUNT(평가 때 수가 적은 육식은 더 많이 둔다).
"""
import csv
import time
from pathlib import Path
from random import Random

from . import config
from .rl import QPolicy, QTable, RulePolicy
from .world import World

LOG_COLUMNS = (["episode", "epsilon"]
               + [f"mean_reward_{s}" for s in config.SPECIES]
               + [f"mean_lifespan_{s}" for s in config.SPECIES]
               + ["hunts"] + [f"hunts_{s}" for s in config.CARNIVORES]
               + [f"q_entries_{s}" for s in config.SPECIES])

HERBIVORES = frozenset(config.HERBIVORES)
CARNIVORES = frozenset(config.CARNIVORES)


def _epsilon(i, n):
    """phase 안에서 EPSILON_START → EPSILON_END 선형 감소."""
    if n <= 1:
        return config.EPSILON_END
    return config.EPSILON_START + (config.EPSILON_END - config.EPSILON_START) * i / (n - 1)


def _alpha(i, n):
    """phase 안에서 ALPHA → ALPHA_END 선형 감소(ε과 같은 일정)."""
    if n <= 1:
        return config.ALPHA_END
    return config.ALPHA + (config.ALPHA_END - config.ALPHA) * i / (n - 1)


def plan(episodes, schedule="alternating"):
    """episode별 (학습하는 종 집합, ε, 육식 상대역이 규칙 기반인가, α) 목록."""
    out = []
    if schedule == "joint":
        for i in range(episodes):
            out.append((set(config.SPECIES), _epsilon(i, episodes), False, _alpha(i, episodes)))
        return out
    if schedule != "alternating":
        raise ValueError("schedule은 'alternating' 또는 'joint'여야 합니다.")
    n1 = int(episodes * 0.4)
    n2 = int(episodes * 0.4)
    for i in range(n1):
        out.append((set(HERBIVORES), _epsilon(i, n1), True, _alpha(i, n1)))
    for i in range(n2):
        out.append((set(CARNIVORES), _epsilon(i, n2), False, _alpha(i, n2)))
    for i in range(episodes - n1 - n2):
        out.append((set(HERBIVORES) if (i // 10) % 2 == 0 else set(CARNIVORES),
                    config.EPSILON_END, False, config.ALPHA_END))
    return out


def train(episodes=None, steps=None, seed=0, schedule="alternating", save=True,
          results_dir=None, out=print):
    episodes = config.TRAIN_EPISODES if episodes is None else episodes
    steps = config.TRAIN_STEPS if steps is None else steps
    tables = {s: QTable(s) for s in config.SPECIES}
    rng = Random(seed)
    learners = {s: QPolicy(tables[s], learn=True, rng=Random(rng.random()))
                for s in config.SPECIES}
    rule = {s: RulePolicy(s, Random(rng.random())) for s in config.CARNIVORES}
    rows = []
    start = time.perf_counter()
    for ep, (learning, eps, carn_is_rule, alpha) in enumerate(plan(episodes, schedule)):
        world = World(seed=seed * 1000003 + ep, mode="train", record_history=False)
        controllers = {}
        for s in config.SPECIES:
            pol = learners[s]
            pol.learn = s in learning
            pol.epsilon = eps if pol.learn else config.EPSILON_END
            pol.alpha = alpha
            controllers[s] = pol
        if carn_is_rule:
            controllers.update(rule)
        for _ in range(steps):
            world.step(controllers)
        # episode가 끝나도 terminal 갱신은 하지 않는다(대기 중인 전이는 버림).
        m = world.metrics
        row = {"episode": ep + 1, "epsilon": round(eps, 4)}
        for s in config.SPECIES:
            n = m[f"agent_steps_{s}"]
            row[f"mean_reward_{s}"] = round(m[f"reward_{s}"] / max(1, n), 5)
            row[f"mean_lifespan_{s}"] = round(n / max(1, m[f"deaths_{s}"]), 2)
            row[f"q_entries_{s}"] = tables[s].entries()
        row["hunts"] = m["hunts"]
        for s in config.CARNIVORES:
            row[f"hunts_{s}"] = m[f"hunts_{s}"]
        rows.append({k: row[k] for k in LOG_COLUMNS})
        if out and ((ep + 1) % max(1, episodes // 15) == 0 or ep == 0 or ep + 1 == episodes):
            who = ("초식" if learning == HERBIVORES else "육식" if learning == CARNIVORES
                   else "전체") + ("(육식=규칙)" if carn_is_rule else "")
            label = config.SPECIES_LABEL
            out(f"episode {ep + 1:>4}/{episodes} [{who:<10}] ε={eps:.3f} 보상 "
                + " ".join(f"{label[s]} {row[f'mean_reward_{s}']:+.3f}" for s in config.SPECIES)
                + " | 수명 "
                + " ".join(f"{label[s]} {row[f'mean_lifespan_{s}']:.0f}" for s in config.SPECIES)
                + " | 사냥 "
                + " ".join(f"{label[s]} {row[f'hunts_{s}']}" for s in config.CARNIVORES)
                + " | Q " + "/".join(str(row[f"q_entries_{s}"]) for s in config.SPECIES)
                + f" | {time.perf_counter() - start:.0f}s")
    elapsed = time.perf_counter() - start
    if save:
        for t in tables.values():
            t.save()
        results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        if out:
            out(f"저장: {config.Q_TABLE_DIR} / {results_dir / 'train_log.csv'}")
    if out:
        out(f"학습 완료: {episodes} episode × {steps} step, {elapsed:.1f}초")
    return tables, rows, elapsed
