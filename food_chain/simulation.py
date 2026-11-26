"""Simulation(뷰어 · 평가 공용 진입점)과 정책 비교 평가.

Simulation은 eval 모드(번식 · 사망 ON, 학습 OFF)로 세계를 한 step씩 진행한다.
"""
import csv
from pathlib import Path
from random import Random

from . import config
from .rl import QTable, make_policy
from .world import World, CAUSES

POLICY_MODES = ("random", "rule", "trained")
NO_PREDATORS = {s: 0 for s in config.CARNIVORES}        # '포식자 없음' 대조 실행용


def stats_keys():
    """stats() / history 원소 / eval CSV가 공유하는 키 순서."""
    keys = ["turn"]
    for s in config.SPECIES:
        keys += [f"count_{s}", f"avg_energy_{s}", f"avg_hydration_{s}",
                 f"births_{s}", f"deaths_{s}"]
    keys += ["grass_ratio", "browse_ratio"]
    keys += [f"deaths_{c}" for c in CAUSES]
    keys += [f"deaths_predation_{s}" for s in config.HERBIVORES]
    keys.append("hunts")
    for s in config.CARNIVORES:
        keys += [f"hunts_{s}", f"immigrants_{s}"]
    keys += [f"sprints_{s}" for s in config.SPRINTERS]
    keys += [f"far_hunts_{s}" for s in config.RANGED_HUNTERS]
    return keys


def load_tables():
    """저장된 종별 Q-Table을 읽는다. 하나라도 없으면 FileNotFoundError."""
    return {s: QTable.load(s) for s in config.SPECIES}


def make_controllers(policy_mode, seed, tables=None):
    """종별 정책을 만든다. policy_mode는 문자열 하나 또는 {종: 정책} dict."""
    modes = policy_mode if isinstance(policy_mode, dict) \
        else {s: policy_mode for s in config.SPECIES}
    for mode in modes.values():
        if mode not in POLICY_MODES:
            raise ValueError("policy_mode는 'trained', 'random', 'rule' 중 하나여야 합니다.")
    if "trained" in modes.values() and tables is None:
        tables = load_tables()
    out = {}
    for i, s in enumerate(config.SPECIES):
        rng = Random(None if seed is None else seed * 1000003 + 17 + i)
        out[s] = make_policy(s, modes[s], rng, tables[s] if tables else None)
    return out


class Simulation:
    def __init__(self, seed=42, policy_mode="trained", tables=None, initial_counts=None):
        if policy_mode not in POLICY_MODES:
            raise ValueError("policy_mode는 'trained', 'random', 'rule' 중 하나여야 합니다.")
        self.policy_mode = policy_mode
        # 'trained'인데 저장된 Q-Table이 없으면 여기서 FileNotFoundError
        self.tables = tables if tables is not None else (
            load_tables() if policy_mode == "trained" else None)
        self.world = World(seed=seed, mode="eval", initial_counts=initial_counts)
        self.controllers = make_controllers(policy_mode, self.world.seed, self.tables)

    def reset(self, seed=None):
        """seed=None이면 새 무작위 배치."""
        self.world.reset(seed)
        self.controllers = make_controllers(self.policy_mode, self.world.seed, self.tables)

    def step(self):
        self.world.step(self.controllers)

    @property
    def seed(self):
        return self.world.seed


# ── 평가 ────────────────────────────────────────────────────────────────
def run_population(policy_mode, seed, steps, tables=None, initial_counts=None):
    """eval 모드(번식 ON)로 steps만큼 돌린 뒤 (history, 요약)을 돌려준다.

    요약의 extinct_step_<종>은 그 종이 0마리가 된 step(멸종하지 않았으면 -1),
    coexist_steps는 처음에 있던 종 가운데 하나라도 멸종하기 전까지의 step 수.
    """
    sim = Simulation(seed=seed, policy_mode=policy_mode, tables=tables,
                     initial_counts=initial_counts)
    w = sim.world
    present = [s for s in config.SPECIES if w.count(s) > 0]
    extinct = {}
    for _ in range(steps):
        sim.step()
        for s in present:
            if s not in extinct and w.n_alive[s] == 0:
                extinct[s] = w.turn
        if not w.animals:
            break
    h = w.history
    m = w.metrics
    summary = {
        "policy": policy_mode, "seed": seed, "steps": w.turn,
        "coexist_steps": min(extinct.values()) if extinct else steps,
        "coexisted": int(not extinct),
    }
    for s in config.SPECIES:
        counts = [x[f"count_{s}"] for x in h]
        summary[f"initial_{s}"] = counts[0]
        summary[f"final_{s}"] = counts[-1]
        summary[f"mean_{s}"] = sum(counts) / len(counts)
        summary[f"min_{s}"] = min(counts)
        summary[f"max_{s}"] = max(counts)
        summary[f"extinct_step_{s}"] = extinct.get(s, -1)
        summary[f"births_{s}"] = m[f"births_{s}"]
        summary[f"zero_steps_{s}"] = sum(1 for c in counts if c == 0)
        summary[f"immigrants_{s}"] = m.get(f"immigrants_{s}", 0)
        summary[f"cap_steps_{s}"] = sum(1 for c in counts
                                        if c >= w.params[s]["population_cap"])
    summary["mean_grass"] = sum(x["grass_ratio"] for x in h) / len(h)
    summary["mean_browse"] = sum(x["browse_ratio"] for x in h) / len(h)
    summary["browse_cells"] = len(w.browse_cells)
    summary["burrow_cells"] = len(w.burrow_cells)
    # 굴 안에 있던 평균 마릿수 = 굴 안에서 보낸 개체·step / 진행 step (굴을 쓰는 종만)
    for s in config.BURROW_USERS:
        summary[f"mean_in_refuge_{s}"] = m[f"shelter_steps_{s}"] / max(1, w.turn)
    for s in config.SPRINTERS:                       # 질주(2칸 이동) 발동 횟수, 그중 위협이 보이던 때
        summary[f"sprints_{s}"] = m[f"sprints_{s}"]
        summary[f"sprints_threat_{s}"] = m[f"sprints_threat_{s}"]
    summary["hunts"] = m["hunts"]
    for c in config.CARNIVORES:
        for p in config.PREY[c]:
            summary[f"kills_{c}_{p}"] = m[f"kills_{c}_{p}"]
    for c in config.RANGED_HUNTERS:                  # 2칸 이상 떨어진 먹이를 잡은 횟수
        for p in config.PREY[c]:
            summary[f"far_kills_{c}_{p}"] = m[f"far_kills_{c}_{p}"]
    for c in CAUSES:
        summary[f"deaths_{c}"] = w.cause_total[c]
    for s in config.SPECIES:
        for c in CAUSES:
            if not (s in config.PREY and c == "predation"):
                summary[f"{s}_{c}"] = m[f"deaths_{s}_{c}"]
    return h, summary


def run_benchmark(policy_mode, seed, steps, tables=None):
    """개체수를 고정(번식 OFF, 죽으면 재생성)하고 정책의 '개체 단위 실력'만 잰다.

    세 정책이 똑같은 밀도(config.TRAIN_COUNT)에서 비교되므로 평균 수명 · 사냥 성공 · 도주율을
    공정하게 볼 수 있다. policy_mode에 {종: 정책} dict를 주면 종별로 다른 정책을 쓴다.
    """
    w = World(seed=seed, mode="train", randomize=False, record_history=False)
    ctrl = make_controllers(policy_mode, seed, tables)
    for _ in range(steps):
        w.step(ctrl)
    m = w.metrics
    out = {
        "policy": policy_mode if isinstance(policy_mode, str)
        else "/".join(policy_mode[s] for s in config.SPECIES),
        "seed": seed,
    }
    for s in config.SPECIES:
        n = m[f"agent_steps_{s}"]
        # 평균 수명 = 개체·step 수 / 사망 수 (사망이 없으면 관측 step 전체)
        out[f"lifespan_{s}"] = n / max(1, m[f"deaths_{s}"])
        out[f"reward_{s}"] = m[f"reward_{s}"] / max(1, n)
        for c in CAUSES:
            out[f"{s}_{c}_per_1000"] = 1000.0 * m[f"deaths_{s}_{c}"] / max(1, n)
    for s in config.HERBIVORES:
        threat = m[f"threat_steps_{s}"]
        out[f"threat_steps_{s}"] = threat
        out[f"flee_rate_{s}"] = m[f"flee_steps_{s}"] / max(1, threat)
        out[f"approach_danger_rate_{s}"] = m[f"approach_danger_steps_{s}"] / max(1, threat)
    for s in config.CARNIVORES:
        n = m[f"agent_steps_{s}"]
        hungry = m[f"hungry_steps_{s}"]
        out[f"hunts_per_1000_steps_{s}"] = 1000.0 * m[f"hunts_{s}"] / max(1, n)
        # 배고픈(사냥 가능한) 상태 1000 step당 사냥 성공 수 = 사냥 효율
        out[f"hunts_per_1000_hungry_steps_{s}"] = 1000.0 * m[f"hunts_{s}"] / max(1, hungry)
        out[f"hungry_share_{s}"] = hungry / max(1, n)
        out[f"chase_steps_{s}"] = m[f"chase_steps_{s}"]
        out[f"chase_close_in_rate_{s}"] = m[f"close_in_steps_{s}"] / max(1, m[f"chase_steps_{s}"])
        for p in config.PREY[s]:
            out[f"kills_{s}_{p}"] = m[f"kills_{s}_{p}"]
    return out


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in r.items()})


def print_population(rows_by_policy, out=print):
    """개체군 시뮬레이션 요약(종별 평균 · 최소 · 최대 · 멸종 seed 수 · 사망 원인)을 출력한다."""
    label = config.SPECIES_LABEL
    for p, rows in rows_by_policy.items():
        n = len(rows)
        out(f"  [{p}] 4종 공존 seed {sum(r['coexisted'] for r in rows)}/{n}, "
            f"첫 멸종까지 평균 {_mean(rows, 'coexist_steps'):.0f} step "
            f"(최소 {min(r['coexist_steps'] for r in rows)}), "
            f"평균 풀 비율 {_mean(rows, 'mean_grass'):.2f}, "
            f"관목 비율 {_mean(rows, 'mean_browse'):.2f}")
        out(f"    {'종':<4} {'평균':>7} {'최소':>5} {'최대':>5} {'최종 평균':>8} "
            f"{'멸종 seed':>9} {'멸종 step':>20}  사망(아사/탈수/포식/노화, seed 평균)")
        for s in config.SPECIES:
            if not any(r[f"initial_{s}"] for r in rows):
                continue
            gone = sorted(r[f"extinct_step_{s}"] for r in rows if r[f"extinct_step_{s}"] >= 0)
            when = ",".join(map(str, gone)) if gone else "-"
            deaths = "/".join(f"{_mean(rows, f'{s}_{c}'):.0f}" if f"{s}_{c}" in rows[0] else "-"
                              for c in CAUSES)
            cap = sum(r[f"cap_steps_{s}"] for r in rows)
            imm = sum(r[f"immigrants_{s}"] for r in rows)
            out(f"    {label[s]:<4} {_mean(rows, f'mean_{s}'):>7.1f} "
                f"{min(r[f'min_{s}'] for r in rows):>5} {max(r[f'max_{s}'] for r in rows):>5} "
                f"{_mean(rows, f'final_{s}'):>8.1f} {len(gone):>6}/{n:<2} {when:>20}  {deaths}"
                + (f"  (cap 도달 {cap} step)" if cap else "")
                + (f"  [외부 유입 {imm}마리, 유입받은 seed "
                   f"{sum(1 for r in rows if r[f'immigrants_{s}'])}/{n}]" if imm else ""))
        hide = "  ".join(f"{label[s]} 평균 {_mean(rows, f'mean_in_refuge_{s}'):.1f}마리"
                         for s in config.BURROW_USERS)
        out(f"    굴({_mean(rows, 'burrow_cells'):.0f}칸) 안: {hide}")
        kills = "  ".join(f"{label[c]}→{label[q]} {_mean(rows, f'kills_{c}_{q}'):.0f}"
                          for c in config.CARNIVORES for q in config.PREY[c])
        out(f"    사냥(seed 평균): {kills}")
        far = "  ".join(f"{label[c]}→{label[q]} {_mean(rows, f'far_kills_{c}_{q}'):.0f}"
                        for c in config.RANGED_HUNTERS for q in config.PREY[c])
        if far:
            out(f"    그중 2칸 거리 사냥(seed 평균): {far}")
        for s in config.SPRINTERS:
            out(f"    {label[s]} 질주 발동(seed 평균): {_mean(rows, f'sprints_{s}'):.0f}회 "
                f"(위협이 보일 때 {_mean(rows, f'sprints_threat_{s}'):.0f}회)")


def evaluate(seeds=None, steps=None, bench_steps=None, tables=None, results_dir=None,
             policies=POLICY_MODES, first_seed=1000, out=print, ablation=False):
    """random / rule / trained를 같은 seed들로 비교하고 CSV를 저장한다.

    ablation=True면 '포식자 없음'(여우 · 늑대 0마리) 대조 실행을 추가한다.
    """
    seeds = config.EVAL_SEEDS if seeds is None else seeds
    steps = config.EVAL_STEPS if steps is None else steps
    bench_steps = config.BENCH_STEPS if bench_steps is None else bench_steps
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_list = list(range(first_seed, first_seed + seeds))
    keys = stats_keys()
    label = config.SPECIES_LABEL
    herb = {s: None for s in config.HERBIVORES}
    pop, bench, cross, nopred = {}, {}, {}, {}

    def write_history(path, runs):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seed"] + keys)
            for seed, history in runs:
                for row in history:
                    writer.writerow([seed] + [round(row[k], 4) if isinstance(row[k], float)
                                              else row[k] for k in keys])

    for policy in policies:
        if policy == "trained" and tables is None:
            try:
                tables = load_tables()
            except FileNotFoundError as err:
                out(f"[건너뜀] trained: {err}")
                continue
        t = tables if policy == "trained" else None
        pop[policy], bench[policy] = [], []
        runs = []
        for seed in seed_list:
            history, summary = run_population(policy, seed, steps, t)
            runs.append((seed, history))
            pop[policy].append(summary)
            bench[policy].append(run_benchmark(policy, seed, bench_steps, t))
        write_history(results_dir / f"eval_{policy}.csv", runs)
        # 상대 고정 비교: 상대 식성을 규칙 기반으로 고정하고 한쪽(초식 / 육식)의 정책만 바꾼다
        for side, group in (("herbivore", config.HERBIVORES), ("carnivore", config.CARNIVORES)):
            modes = {s: (policy if s in group else "rule") for s in config.SPECIES}
            cross[f"{side}={policy}"] = [dict(run_benchmark(modes, seed, bench_steps, t),
                                              side=side, side_policy=policy)
                                         for seed in seed_list]
        if ablation and policy != "random":
            nopred[policy] = []
            runs = []
            for seed in seed_list:
                history, summary = run_population(policy, seed, steps, t, NO_PREDATORS)
                summary["policy"] = f"{policy}_nopred"
                runs.append((seed, history))
                nopred[policy].append(summary)
            write_history(results_dir / f"eval_{policy}_nopred.csv", runs)

    _write_csv(results_dir / "eval_summary.csv", [r for p in pop for r in pop[p]])
    _write_csv(results_dir / "eval_benchmark.csv", [r for p in bench for r in bench[p]])
    _write_csv(results_dir / "eval_crossplay.csv", [r for p in cross for r in cross[p]])
    if nopred:
        _write_csv(results_dir / "eval_ablation.csv", [r for p in nopred for r in nopred[p]])

    out(f"\n[1] 개체군 시뮬레이션 (eval 모드: 번식 · 사망 ON) — seed {seeds}개 × {steps} step")
    print_population(pop, out)
    if nopred:
        out(f"\n[1-b] 대조: 포식자 없음(여우 · 늑대 0마리) — seed {seeds}개 × {steps} step")
        print_population(nopred, out)

    out(f"\n[2] 고정 개체수 벤치마크 (번식 OFF, 죽으면 재생성) — seed {seeds}개 × {bench_steps} step")
    out("  종별 평균 수명 / step당 보상 / 1000 개체·step당 사망(아사/탈수/포식/노화):")
    for p, rows in bench.items():
        for s in config.SPECIES:
            d = "/".join(f"{_mean(rows, f'{s}_{c}_per_1000'):.2f}" for c in CAUSES)
            out(f"  {p:<8} {label[s]:<3} 수명 {_mean(rows, f'lifespan_{s}'):>7.1f}  "
                f"보상 {_mean(rows, f'reward_{s}'):+.4f}  사망 {d}")
    out("  초식: 가까운 위협 앞에서의 도주율 / 위협 접근율   육식: 사냥 효율 / 추적 접근율")
    for p, rows in bench.items():
        hs = "  ".join(f"{label[s]} 도주 {_mean(rows, f'flee_rate_{s}'):.3f} "
                       f"접근 {_mean(rows, f'approach_danger_rate_{s}'):.3f}" for s in herb)
        cs = "  ".join(f"{label[s]} 사냥/1000배고픈step "
                       f"{_mean(rows, f'hunts_per_1000_hungry_steps_{s}'):.1f} "
                       f"추적 접근 {_mean(rows, f'chase_close_in_rate_{s}'):.3f}"
                       for s in config.CARNIVORES)
        out(f"  {p:<8} {hs} | {cs}")

    out("\n[3] 상대 고정 비교 (같은 벤치마크, 상대 식성은 규칙 기반으로 고정)")
    out("  초식 정책만 교체 (육식 = rule):")
    for p in bench:
        rows = cross[f"herbivore={p}"]
        for s in config.HERBIVORES:
            d = "/".join(f"{_mean(rows, f'{s}_{c}_per_1000'):.2f}" for c in CAUSES)
            out(f"    {label[s]} {p:<8} 수명 {_mean(rows, f'lifespan_{s}'):>6.1f}  "
                f"도주율 {_mean(rows, f'flee_rate_{s}'):.3f}  "
                f"step당 보상 {_mean(rows, f'reward_{s}'):+.4f}  사망/1000step {d}")
    out("  육식 정책만 교체 (초식 = rule):")
    for p in bench:
        rows = cross[f"carnivore={p}"]
        for s in config.CARNIVORES:
            d = "/".join(f"{_mean(rows, f'{s}_{c}_per_1000'):.2f}"
                         for c in ("starvation", "dehydration", "age"))
            out(f"    {label[s]} {p:<8} 수명 {_mean(rows, f'lifespan_{s}'):>7.1f}  "
                f"사냥/1000배고픈step {_mean(rows, f'hunts_per_1000_hungry_steps_{s}'):>5.1f}  "
                f"배고픈 시간 비율 {_mean(rows, f'hungry_share_{s}'):.3f}  "
                f"추적 접근율 {_mean(rows, f'chase_close_in_rate_{s}'):.3f}  "
                f"step당 보상 {_mean(rows, f'reward_{s}'):+.4f}  사망/1000step(아사/탈수/노화) {d}")
    out(f"\nCSV 저장: {results_dir}")
    return pop, bench, cross
