"""Q-Table, 학습기(SARSA / Q-Learning), 비교용 정책(random / rule).

종별 Q-Table 1개를 같은 종의 모든 개체가 공유한다.
    초식(토끼 · 사슴) = SARSA      : Q[s,a] += α(r + γ·Q[s',a'] − Q[s,a])   (a' = 다음에 실제로 실행할 행동)
    육식(여우 · 늑대) = Q-Learning : Q[s,a] += α(r + γ·max Q[s',·] − Q[s,a])
갱신은 "지연 방식"이다. 개체가 다음 자기 차례에 s'를 관측하고 a'를 고른 순간,
직전 전이 (s, a, 그 사이 누적 보상 r)을 갱신한다. 그 전에 죽으면 died()에서 terminal 갱신을 한다.
"""
import json
from pathlib import Path
from random import Random

from . import config

N_ACTIONS = 5
# 상태 튜플 각 성분의 크기. 종 이름이 아니라 역할(초식 / 육식 / 먹이 크기를 구분하는 육식)로 정한다.
HERBIVORE_SHAPE = (3, 3, 5, 5, 5, 3)      # (energy, hydration, 풀 방향, 물 방향, 위험 방향, 위험 거리)
# '풀 방향'은 그 종의 먹이 방향이다: 토끼 = 풀, 사슴 = 관목(config.DIET). 표 크기는 그대로다.
# 질주하는 초식(사슴): + 달리는 방향(0 = 위협이 안 보이거나 직전에 멈춤, 1~4 = 직전 이동 방향)
HERBIVORE_SPRINT_SHAPE = HERBIVORE_SHAPE + (5,)
CARNIVORE_SHAPE = (3, 3, 5, 3, 5)         # (energy, hydration, 먹이 방향, 먹이 거리, 물 방향)
CARNIVORE_SIZED_SHAPE = CARNIVORE_SHAPE + (3,)   # + 먹이 크기(0 없음, 1 소형, 2 대형)


def _shape(species):
    if species not in config.PREY:
        return HERBIVORE_SPRINT_SHAPE if species in config.SPRINTERS else HERBIVORE_SHAPE
    sizes = {config.PREY_SIZE[p] for p in config.PREY[species]}
    return CARNIVORE_SIZED_SHAPE if len(sizes) > 1 else CARNIVORE_SHAPE


STATE_SHAPE = {s: _shape(s) for s in config.SPECIES}
OPPOSITE = (0, 2, 1, 4, 3)
PERPENDICULAR = ((), (3, 4), (3, 4), (1, 2), (1, 2))


def n_states(species):
    n = 1
    for k in STATE_SHAPE[species]:
        n *= k
    return n


def encode_herbivore(o):
    return ((((o[0] * 3 + o[1]) * 5 + o[2]) * 5 + o[3]) * 5 + o[4]) * 3 + o[5]


def encode_herbivore_sprint(o):
    return (((((o[0] * 3 + o[1]) * 5 + o[2]) * 5 + o[3]) * 5 + o[4]) * 3 + o[5]) * 5 + o[6]


def encode_carnivore(o):
    return (((o[0] * 3 + o[1]) * 5 + o[2]) * 3 + o[3]) * 5 + o[4]


def encode_carnivore_sized(o):
    return ((((o[0] * 3 + o[1]) * 5 + o[2]) * 3 + o[3]) * 5 + o[4]) * 3 + o[5]


_ENCODER_OF_SHAPE = {HERBIVORE_SHAPE: encode_herbivore,
                     HERBIVORE_SPRINT_SHAPE: encode_herbivore_sprint,
                     CARNIVORE_SHAPE: encode_carnivore,
                     CARNIVORE_SIZED_SHAPE: encode_carnivore_sized}
ENCODERS = {s: _ENCODER_OF_SHAPE[STATE_SHAPE[s]] for s in config.SPECIES}
encode_rabbit = encode_herbivore
encode_deer = ENCODERS["deer"]
encode_fox = encode_carnivore
encode_wolf = encode_carnivore_sized


def decode(species, index):
    out = []
    for k in reversed(STATE_SHAPE[species]):
        out.append(index % k)
        index //= k
    return tuple(reversed(out))


class QTable:
    def __init__(self, species, algorithm=None):
        self.species = species
        self.algorithm = algorithm or config.ALGORITHM[species]
        self.q = [[0.0] * N_ACTIONS for _ in range(n_states(species))]
        # 갱신 횟수(학습 중에만 사용, 저장하지 않음)
        self.visits = [[0] * N_ACTIONS for _ in range(n_states(species))]

    def entries(self):
        """한 번이라도 갱신된 (state, action) 수."""
        return sum(1 for row in self.q for v in row if v != 0.0)

    def greedy(self, state, rng):
        row = self.q[state]
        best = max(row)
        if row.count(best) == 1:
            return row.index(best)
        return rng.choice([i for i, v in enumerate(row) if v == best])

    def select(self, state, epsilon, rng):
        """ε-greedy. 최댓값이 여럿이면 그중 무작위."""
        if epsilon > 0.0 and rng.random() < epsilon:
            return rng.randrange(N_ACTIONS)
        return self.greedy(state, rng)

    def update_sarsa(self, s, a, r, s2, a2, alpha, gamma):
        row = self.q[s]
        row[a] += alpha * (r + gamma * self.q[s2][a2] - row[a])

    def update_qlearning(self, s, a, r, s2, alpha, gamma):
        row = self.q[s]
        row[a] += alpha * (r + gamma * max(self.q[s2]) - row[a])

    def update_terminal(self, s, a, r, alpha):
        row = self.q[s]
        row[a] += alpha * (r - row[a])

    def save(self, path=None):
        path = Path(path) if path else config.Q_TABLE_DIR / f"{self.species}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"species": self.species, "algorithm": self.algorithm,
                "state_shape": list(STATE_SHAPE[self.species]),
                "actions": list(config.ACTIONS), "vision": config.VISION[self.species],
                "q": [[round(v, 6) for v in row] for row in self.q]}
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @classmethod
    def load(cls, species, path=None):
        path = Path(path) if path else config.Q_TABLE_DIR / f"{species}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"학습된 Q-Table이 없습니다: {path} (먼저 python -m food_chain --train 실행)")
        data = json.loads(path.read_text(encoding="utf-8"))
        if tuple(data["state_shape"]) != STATE_SHAPE[species]:
            raise ValueError(f"{path}: 상태 구성이 현재 코드와 다릅니다. 다시 학습하세요.")
        table = cls(species, data.get("algorithm"))
        table.q = [list(map(float, row)) for row in data["q"]]
        return table


# ── 정책 ────────────────────────────────────────────────────────────────
class RandomPolicy:
    """5개 행동 중 균등 무작위."""

    def __init__(self, rng=None):
        self.rng = rng or Random()

    def act(self, world, a, obs):
        return self.rng.randrange(N_ACTIONS)

    def died(self, world, a):
        pass


class QPolicy:
    """Q-Table 정책. learn=True면 지연 갱신으로 학습한다."""

    def __init__(self, table, epsilon=0.0, learn=False, rng=None,
                 alpha=None, gamma=None):
        self.table = table
        self.encode = ENCODERS[table.species]
        self.sarsa = table.algorithm == "sarsa"
        self.epsilon = epsilon
        self.learn = learn
        self.rng = rng or Random()
        self.alpha = config.ALPHA if alpha is None else alpha
        self.gamma = config.GAMMA if gamma is None else gamma
        self.sample_average = config.ALPHA_SAMPLE_AVERAGE

    def _alpha(self, s, a):
        """(s, a)의 학습률. sample_average면 처음 1/α번은 표본 평균(1/n)으로 갱신한다."""
        if not self.sample_average:
            return self.alpha
        row = self.table.visits[s]
        row[a] += 1
        return max(self.alpha, 1.0 / row[a])

    def act(self, world, a, obs):
        s = self.encode(obs)
        action = self.table.select(s, self.epsilon, self.rng)
        if self.learn:
            if a.prev_state >= 0:
                alpha = self._alpha(a.prev_state, a.prev_action)
                if self.sarsa:
                    self.table.update_sarsa(a.prev_state, a.prev_action, a.reward_acc,
                                            s, action, alpha, self.gamma)
                else:
                    self.table.update_qlearning(a.prev_state, a.prev_action, a.reward_acc,
                                                s, alpha, self.gamma)
            a.prev_state = s
            a.prev_action = action
            a.reward_acc = 0.0
        return action

    def died(self, world, a):
        """사망 = terminal. 자기 차례든 남의 차례(포식)든 정확히 1회 호출된다."""
        if self.learn and a.prev_state >= 0:
            self.table.update_terminal(a.prev_state, a.prev_action, a.reward_acc,
                                       self._alpha(a.prev_state, a.prev_action))
            a.prev_state = -1


class RulePolicy:
    """사람이 짠 규칙 기반 정책(비교 기준 + 교대 학습의 상대역).

    초식: 가까운 위협 회피(굴이 보이면 굴로) > 목마르면 물 > 배고프면 풀 > 여유 있을 때 물 · 풀
          질주하는 종(사슴)은 도주할 때 달리던 방향을 유지한다(같은 방향 연속 이동 → 2칸 질주).
    육식: 목마르면 물 > 배고프면 먹이 추적 > 여유 있을 때 물 > 배회
    학습 정책과 같은 관측(이산 상태)에 더해 '그 방향이 막혔는가'만 추가로 본다.
    """

    def __init__(self, species, rng=None):
        self.species = species
        self.carnivore = species in config.PREY
        self.eat_below = config.PARAMS[species].get("eat_below", 0.0)
        self.sprinter = bool(config.PARAMS[species].get("sprint_prob"))
        self.rng = rng or Random()

    def died(self, world, a):
        pass

    def _go(self, world, a, d):
        """d 방향으로 가되 막혀 있으면 옆으로 비켜 간다."""
        if world.can_move(a, d):
            return d
        side = [x for x in PERPENDICULAR[d] if world.can_move(a, x)]
        return self.rng.choice(side) if side else 0

    def _wander(self, world, a):
        free = [x for x in (1, 2, 3, 4) if world.can_move(a, x)]
        return self.rng.choice(free) if free else 0

    def act(self, world, a, obs):
        if self.carnivore:
            e, h, prey_dir, prey_dist, water_dir = obs[:5]
            if h == 0 and water_dir:
                return self._go(world, a, water_dir)
            if e < 2 and prey_dir:
                return self._go(world, a, prey_dir)
            if h < 2 and water_dir:
                return self._go(world, a, water_dir)
            if e == 2 and self.rng.random() < 0.5:
                return 0                                   # 배부르면 느긋하게 배회
            return self._wander(world, a)
        e, h, grass_dir, water_dir, danger_dir, danger_dist = obs[:6]
        if a.refuge >= 0:                                  # 굴이 보이면 굴로(이미 안이면 머문다)
            return self._go(world, a, grass_dir) if grass_dir else 0
        if danger_dist == 1:
            return self._flee(world, a)
        if h == 0 and water_dir:
            return self._go(world, a, water_dir)
        if e < 2 and grass_dir:
            return self._go(world, a, grass_dir)
        if h < 2 and water_dir:
            return self._go(world, a, water_dir)
        if a.energy < self.eat_below:
            return self._go(world, a, grass_dir) if grass_dir else self._wander(world, a)
        return 0

    def _flee(self, world, a):
        t = a.target
        best, moves = -1, [0]
        for x in range(N_ACTIONS):
            if not world.can_move(a, x):
                continue
            dr, dc = config.MOVES[x]
            d = abs(t.row - a.row - dr) + abs(t.col - a.col - dc)
            if d > best:
                best, moves = d, [x]
            elif d == best:
                moves.append(x)
        # 질주하는 종: 거리를 똑같이 벌리는 방향이 여럿이면 달리던 방향을 유지한다(직선 질주).
        if self.sprinter and a.last_dir in moves:
            return a.last_dir
        return self.rng.choice(moves)


def make_policy(species, mode, rng=None, table=None):
    """mode: 'random' | 'rule' | 'trained'."""
    if mode == "random":
        return RandomPolicy(rng)
    if mode == "rule":
        return RulePolicy(species, rng)
    if mode == "trained":
        return QPolicy(table or QTable.load(species), epsilon=config.EVAL_EPSILON,
                       learn=False, rng=rng)
    raise ValueError("policy_mode는 'trained', 'random', 'rule' 중 하나여야 합니다.")
