"""격자 세계: 풀 · 관목 · 물웅덩이 · 굴 · 초식(토끼 · 사슴) · 육식(여우 · 늑대).

표준 라이브러리만 사용한다. 한 step에서 모든 개체가 무작위 순서로 한 번씩 행동하고,
먹기 · 마시기 · 사냥 · 번식 · 사망은 환경이 자동으로 처리한다.
종 사이의 관계(누가 누구를 잡는가, 종별 수치)는 전부 config에서 읽으므로
종이 늘어도 이 파일의 구조는 그대로다(역할 = 초식 / 육식으로만 분기한다).

정책(controller)은 종마다 하나씩 넘기며 다음 두 메서드를 가진다.
    act(world, animal, obs) -> int      # 행동 선택(학습기는 여기서 직전 전이를 갱신)
    died(world, animal) -> None         # 사망 직후 1회 호출(학습기의 terminal 갱신)
보상은 animal.reward_acc에 누적된다(다른 개체 차례에 잡아먹힌 경우의 -5 포함).
"""
from collections import deque
from random import Random, SystemRandom

from . import config

CAUSES = ("starvation", "dehydration", "predation", "age")


def direction(dy, dx):
    """상대 위치 → 방향 코드. 0 없음, 1 위, 2 아래, 3 왼쪽, 4 오른쪽 (|dy| ≥ |dx|이면 세로 우선)."""
    if dy == 0 and dx == 0:
        return 0
    if abs(dy) >= abs(dx):
        return 1 if dy < 0 else 2
    return 3 if dx < 0 else 4


def level_bin(value):
    """energy · hydration 3구간: 0~34 → 0, 35~69 → 1, 70~100 → 2."""
    return 0 if value < 35 else (1 if value < 70 else 2)


def diamond(radius):
    """맨해튼 반경 안의 (dy, dx, 거리, 방향) 목록. 자기 칸 제외, 가까운 순."""
    cells = []
    for dy in range(-radius, radius + 1):
        width = radius - abs(dy)
        for dx in range(-width, width + 1):
            if dy or dx:
                cells.append((dy, dx, abs(dy) + abs(dx), direction(dy, dx)))
    cells.sort(key=lambda t: t[2])
    return cells


def shuffled_rings(cells, rng):
    """diamond() 결과를 거리 순서는 유지한 채 같은 거리끼리만 섞는다."""
    rings = {}
    for cell in cells:
        rings.setdefault(cell[2], []).append(cell)
    out = []
    for dist in sorted(rings):
        rng.shuffle(rings[dist])
        out.extend(rings[dist])
    return out


class Animal:
    __slots__ = ("id", "species", "carnivore", "eats", "sizes", "size", "row", "col",
                 "energy", "hydration", "age", "lifespan", "cooldown", "alive", "death_cause",
                 "steps_alive", "prev_state", "prev_action", "reward_acc",
                 "target", "target_dist", "burrower", "refuge", "refuge_dist",
                 "last_dir", "streak", "sprinter", "browser")

    def __init__(self, id, species, row, col, energy, hydration, age, lifespan, cooldown):
        self.id = id
        self.species = species
        self.carnivore = species in config.PREY
        self.eats = frozenset(config.PREY.get(species, ()))      # 잡을 수 있는 종
        # 먹이 크기가 여러 가지인 포식자(늑대)만 State에 '먹이 크기'를 넣는다
        self.sizes = len({config.PREY_SIZE[s] for s in self.eats})
        self.size = config.PREY_SIZE.get(species, 0)             # 먹이로서의 크기(육식은 0)
        self.row, self.col = row, col
        self.energy, self.hydration = energy, hydration
        self.age, self.lifespan, self.cooldown = age, lifespan, cooldown
        self.alive = True
        self.death_cause = ""
        self.steps_alive = 0
        self.prev_state = -1       # 학습기가 쓰는 직전 (state, action)
        self.prev_action = 0
        self.reward_acc = 0.0      # 직전 행동 이후 누적 보상
        self.target = None         # 관측 때 본 가장 가까운 위협(초식) / 먹이(육식)
        self.target_dist = 0
        self.burrower = species in config.BURROW_USERS           # 굴에 들어갈 수 있는가
        self.refuge = -1           # 관측 때 State에 넣은 굴 칸(없으면 -1)
        self.refuge_dist = 0
        self.last_dir = 0          # 직전 step에 실제로 이동한 방향(이동하지 않았으면 0)
        self.streak = 0            # 그 방향으로 연속 이동한 횟수(질주 조건)
        self.sprinter = species in config.SPRINTERS              # State에 '달리는 방향'이 붙는 종
        # 관목(잎 · 새순)을 먹는 종(browser)인가. 아니면 풀을 먹는다(grazer). 육식은 항상 False.
        self.browser = species in config.BROWSERS

    def __repr__(self):
        return (f"Animal({self.id}, {self.species}, ({self.row},{self.col}), "
                f"E={self.energy:.0f}, H={self.hydration:.0f}, age={self.age})")


class World:
    def __init__(self, seed=None, mode="eval", randomize=None, record_history=True,
                 initial_counts=None):
        """initial_counts: {'fox': 0, 'wolf': 0}처럼 종별 초기 개체수를 덮어쓴다(대조 실험용)."""
        if mode not in ("train", "eval"):
            raise ValueError("mode는 'train' 또는 'eval'이어야 합니다.")
        self.mode = mode
        # train 모드에서는 기본적으로 episode마다 개체수 · 풀 양을 무작위화한다.
        self.randomize = (mode == "train") if randomize is None else randomize
        self.record_history = record_history
        self.initial_counts = dict(initial_counts or {})
        self.rows, self.cols = config.ROWS, config.COLS
        self.vision = dict(config.VISION)
        self.params = {s: dict(config.PARAMS[s]) for s in config.SPECIES}
        self.reset(seed)

    # ── 초기화 ──────────────────────────────────────────────────────────
    def reset(self, seed=None):
        """seed=None이면 새 무작위 배치."""
        self.seed = SystemRandom().randrange(2 ** 32) if seed is None else seed
        self.rng = rng = Random(self.seed)
        self.turn = 0
        self.next_id = 1
        R, C = self.rows, self.cols
        n = R * C

        self.regrow = float(config.GRASS_REGROW_STEPS)
        counts = {s: self.params[s]["initial_count"] for s in config.SPECIES}
        if self.mode == "train":
            counts.update(config.TRAIN_COUNT)
        counts.update(self.initial_counts)
        if self.randomize:
            lo, hi = config.TRAIN_COUNT_RANGE
            counts = {s: (max(1, round(k * rng.uniform(lo, hi))) if k else 0)
                      for s, k in counts.items()}
            self.regrow *= rng.uniform(*config.TRAIN_REGROW_RANGE)
        self.edible_dt = self.regrow * config.GRASS_EDIBLE_LEVEL
        # 관목: 풀과 같은 배율로 무작위화한다(풍년 · 흉년은 함께 온다)
        self.browse_regrow = float(config.BROWSE_REGROW_STEPS) * self.regrow \
            / config.GRASS_REGROW_STEPS
        self.browse_edible_dt = self.browse_regrow * config.BROWSE_EDIBLE_LEVEL

        self._make_terrain(rng)
        # 풀: eaten[i] = 마지막으로 뜯긴 시각. 수준 = (turn - eaten) / regrow, 0~1.
        full = rng.uniform(0.3, 1.0) if self.randomize else 0.6   # 처음부터 다 자란 칸의 비율
        self.eaten = [0.0] * n
        for i in self.meadow_cells:
            lvl = 1.0 if rng.random() < full else rng.random()
            self.eaten[i] = -lvl * self.regrow
        # 관목: browse_eaten[i] = 마지막으로 뜯긴 시각(풀과 같은 방식, 재생 시간만 다르다).
        self.browse_eaten = [0.0] * n
        for i in self.browse_cells:
            lvl = 1.0 if rng.random() < full else rng.random()
            self.browse_eaten[i] = -lvl * self.browse_regrow

        self.occ = [None] * n
        self.animals = []
        self.n_alive = {s: 0 for s in config.SPECIES}
        # hunters[피식 종] = 그 종을 잡을 수 있는 포식자 목록(초식의 위협 탐색용)
        self.hunters = {h: [] for h in config.HERBIVORES}
        # 같은 거리의 칸들은 순서를 섞은 변형을 여러 벌 만들어 두고 칸마다 다른 것을 쓴다
        # (가까운 것부터 찾되, 동률일 때 특정 방향으로 쏠리지 않게). 시야 반경별로 하나씩.
        local = Random(self.seed * 7919 + 1)
        self._offsets = {v: [shuffled_rings(diamond(v), local) for _ in range(8)]
                         for v in sorted(set(self.vision.values()))}
        self._grass_nb = {v: [None] * n for v in self._offsets}
        self._browse_nb = {v: [None] * n for v in self._offsets}
        self._prey_nb = {v: [None] * n for v in self._offsets}
        self._refuge_nb = {v: [None] * n for v in self._offsets}
        self.near = dict(config.NEAR_DIST)                     # 종별 '가까움' 최대 거리
        # 사냥 반경 안의 칸 (dr, dc, 거리). 반경 1이면 상하좌우 4칸.
        self._reach = {}
        for c in config.CARNIVORES:
            reach = max(1, int(self.params[c].get("hunt_range", 1)))
            self._reach[c] = tuple((dr, dc, 1) for dr, dc in config.MOVES[1:]) + tuple(
                (dy, dx, dist) for dy, dx, dist, _ in diamond(reach) if dist > 1)
        self._hunt = {c: {p: config.HUNT[(c, p)] for p in config.PREY[c]}
                      for c in config.CARNIVORES}

        # 통계
        self.births = {s: 0 for s in config.SPECIES}          # 이번 step
        self.deaths = {s: 0 for s in config.SPECIES}          # 이번 step
        self.hunts = 0                                         # 이번 step(전 포식 종 합계)
        self.hunts_by = {c: 0 for c in config.CARNIVORES}      # 이번 step(포식 종별)
        self.cause_total = {c: 0 for c in CAUSES}              # 누적(전 종)
        self.metrics = self._blank_metrics()                   # 누적(행동 분석용)

        for s in config.SPECIES:
            for _ in range(counts[s]):
                self._spawn(s, initial=True)
        self.target_counts = counts
        self.history = [self.stats()] if self.record_history else []

    @staticmethod
    def _blank_metrics():
        m = {"hunts": 0, "blocked_moves": 0}
        for s in config.SPECIES:
            m[f"agent_steps_{s}"] = 0      # 개체·step 수
            m[f"reward_{s}"] = 0.0         # 보상 합
            m[f"deaths_{s}"] = 0           # 사망 수
            m[f"life_sum_{s}"] = 0         # 죽은 개체의 생존 step 합
            m[f"births_{s}"] = 0
            for c in CAUSES:
                m[f"deaths_{s}_{c}"] = 0
        for s in config.HERBIVORES:
            m[f"threat_steps_{s}"] = 0             # 가까운(NEAR_DIST 이내) 위협이 있던 step
            m[f"flee_steps_{s}"] = 0               # 그중 거리를 늘린 step
            m[f"approach_danger_steps_{s}"] = 0    # 그중 거리를 줄인 step
            m[f"refuge_steps_{s}"] = 0             # 굴이 State에 보일 때 굴 쪽으로 갔거나 굴 안에 머문 step
            m[f"shelter_steps_{s}"] = 0            # 굴 안에서 보낸 개체·step(굴을 쓰는 종만 증가)
            m[f"sprints_{s}"] = 0                  # 질주(2칸 이동) 발동 횟수
            m[f"sprints_threat_{s}"] = 0           # 그중 위협이 보이던 때
        for s in config.CARNIVORES:
            m[f"hunts_{s}"] = 0
            m[f"hungry_steps_{s}"] = 0             # 사냥 가능한(배고픈) step
            m[f"forage_steps_{s}"] = 0             # 잡식 먹이로 연명한 step
            m[f"immigrants_{s}"] = 0               # 외부에서 유입된 개체 수
            m[f"chase_steps_{s}"] = 0              # 그중 먹이가 보이던 step
            m[f"close_in_steps_{s}"] = 0           # 그중 거리를 줄였거나 잡은 step
            m[f"far_hunts_{s}"] = 0                # 2칸 이상 떨어진 먹이를 잡은 횟수
            for p in config.PREY[s]:
                m[f"kills_{s}_{p}"] = 0
                m[f"far_kills_{s}_{p}"] = 0
        return m

    def clear_animals(self):
        """모든 개체를 없앤다(테스트 · 실험용)."""
        for a in self.animals:
            self.occ[a.row * self.cols + a.col] = None
            a.alive = False
        self.animals = []
        self.n_alive = {s: 0 for s in config.SPECIES}
        self.hunters = {h: [] for h in config.HERBIVORES}

    def _make_terrain(self, rng):
        R, C = self.rows, self.cols
        n = R * C
        # 1) 물웅덩이: 구역마다 1개 + 물이 먼 곳에 보충
        water = [False] * n
        gr, gc = config.POND_GRID
        for i in range(gr):
            for j in range(gc):
                cy = int((i + 0.5) * R / gr) + rng.randint(-3, 3)
                cx = int((j + 0.5) * C / gc) + rng.randint(-4, 4)
                self._add_pond(water, cy, cx, rng.uniform(*config.POND_RADIUS_ROWS),
                               rng.uniform(*config.POND_RADIUS_COLS))
        while True:
            dist = self._shore_bfs(water)
            far = max(range(n), key=lambda k: -1 if water[k] else dist[k])
            if dist[far] <= config.WATER_MAX_DIST:
                break
            self._add_pond(water, far // C, far % C, 1.0, 1.0)
        self.water = water
        self.shore_dist = dist
        self.shore = [(not water[k]) and dist[k] == 0 for k in range(n)]
        # 물가 최단 방향(물가에 있으면 0)
        # water_opts[k]: 물가로 가는 걸음 후보 (방향, 도착 칸). 거리가 줄어드는 칸이 먼저,
        # 그다음 거리가 같은 칸(앞이 다른 동물로 막혔을 때 옆으로 돌아가는 용도).
        wdir = [0] * n
        wopts = [()] * n
        for k in range(n):
            if water[k] or dist[k] <= 0:
                continue
            r, c = divmod(k, C)
            closer, same = [], []
            for a in (1, 2, 3, 4):
                dr, dc = config.MOVES[a]
                rr, cc = r + dr, c + dc
                if 0 <= rr < R and 0 <= cc < C and not water[rr * C + cc]:
                    if dist[rr * C + cc] == dist[k] - 1:
                        closer.append((a, rr * C + cc))
                    elif dist[rr * C + cc] == dist[k]:
                        same.append((a, rr * C + cc))
            rng.shuffle(closer)
            rng.shuffle(same)
            wdir[k] = closer[0][0]
            wopts[k] = tuple(closer + same)
        self.water_dir = wdir
        self.water_opts = wopts
        # 2) 초원: 타원 군집을 목표 면적이 찰 때까지 추가 (plant_grid_400.py의 군집 배치 방식)
        meadow = [False] * n
        covered, target = 0, config.MEADOW_COVERAGE * n
        for _ in range(100):
            if covered >= target:
                break
            cy, cx = rng.randrange(R), rng.randrange(C)
            ry, rx = rng.uniform(4, 7), rng.uniform(6, 10)
            for r in range(max(0, int(cy - ry)), min(R, int(cy + ry) + 1)):
                for c in range(max(0, int(cx - rx)), min(C, int(cx + rx) + 1)):
                    k = r * C + c
                    if not meadow[k] and not water[k] \
                            and ((r - cy) / ry) ** 2 + ((c - cx) / rx) ** 2 <= 1:
                        meadow[k] = True
                        covered += 1
        # 3) 굴: 초원 안에 작은 군집(2×2)으로 흩어 놓는다. 물 · 물가에는 두지 않고, 굴 칸에는 풀이 없다.
        burrow = [False] * n
        land = n - sum(water)
        target = round(config.BURROW_FRACTION * land)
        side = max(1, round(config.BURROW_CLUSTER ** 0.5))
        placed = 0
        candidates = [k for k in range(n) if meadow[k]]
        for _ in range(2000):
            if placed >= target or not candidates:
                break
            k0 = candidates[rng.randrange(len(candidates))]
            r0, c0 = divmod(k0, C)
            for r in range(r0, min(R, r0 + side)):
                for c in range(c0, min(C, c0 + side)):
                    k = r * C + c
                    if placed < target and not water[k] and dist[k] > 0 and not burrow[k]:
                        burrow[k] = True
                        meadow[k] = False
                        placed += 1
        # 4) 관목(사슴의 먹이): 풀밭 · 물 · 굴이 아닌 땅에 타원 군집으로 놓는다(숲 가장자리의 관목대).
        #    BROWSE_ENABLED가 꺼져 있으면 만들지 않는다(난수도 쓰지 않아 지형이 예전과 같다).
        browse = [False] * n
        if config.BROWSE_ENABLED and config.BROWSE_COVERAGE > 0:
            covered, target = 0, config.BROWSE_COVERAGE * n
            for _ in range(400):
                if covered >= target:
                    break
                cy, cx = rng.randrange(R), rng.randrange(C)
                ry, rx = rng.uniform(3, 5), rng.uniform(4, 7)
                for r in range(max(0, int(cy - ry)), min(R, int(cy + ry) + 1)):
                    for c in range(max(0, int(cx - rx)), min(C, int(cx + rx) + 1)):
                        k = r * C + c
                        if not browse[k] and not meadow[k] and not water[k] and not burrow[k] \
                                and ((r - cy) / ry) ** 2 + ((c - cx) / rx) ** 2 <= 1:
                            browse[k] = True
                            covered += 1
        self.browse = browse
        self.browse_cells = [k for k in range(n) if browse[k]]
        self.burrow = burrow
        self.burrow_cells = [k for k in range(n) if burrow[k]]
        self.meadow = meadow
        self.meadow_cells = [k for k in range(n) if meadow[k]]
        self.land_cells = [k for k in range(n) if not water[k]]
        self.open_cells = [k for k in range(n) if not water[k] and not burrow[k]]
        self.edge_cells = [k for k in self.open_cells
                           if k // C in (0, R - 1) or k % C in (0, C - 1)]

    def _add_pond(self, water, cy, cx, ry, rx):
        R, C = self.rows, self.cols
        my, mx = int(ry) + 1, int(rx) + 1               # 가장자리와 한 칸은 띄운다(통로 확보)
        cy = min(max(cy, my), R - 1 - my)
        cx = min(max(cx, mx), C - 1 - mx)
        for r in range(cy - my, cy + my + 1):
            for c in range(cx - mx, cx + mx + 1):
                if ((r - cy) / ry) ** 2 + ((c - cx) / rx) ** 2 <= 1:
                    water[r * C + c] = True

    def _shore_bfs(self, water):
        """각 땅 칸에서 물가(물에 상하좌우로 인접한 땅 칸)까지의 최단 이동 거리."""
        R, C = self.rows, self.cols
        dist = [-1] * (R * C)
        queue = deque()
        for k in range(R * C):
            if water[k]:
                continue
            r, c = divmod(k, C)
            if (r > 0 and water[k - C]) or (r < R - 1 and water[k + C]) \
                    or (c > 0 and water[k - 1]) or (c < C - 1 and water[k + 1]):
                dist[k] = 0
                queue.append(k)
        while queue:
            k = queue.popleft()
            r, c = divmod(k, C)
            for dr, dc in config.MOVES[1:]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < R and 0 <= cc < C:
                    j = rr * C + cc
                    if not water[j] and dist[j] < 0:
                        dist[j] = dist[k] + 1
                        queue.append(j)
        # 물이 하나도 없거나 닿을 수 없는 칸은 아주 먼 것으로 취급
        return [d if d >= 0 or water[k] else 10 ** 6 for k, d in enumerate(dist)]

    # ── 개체 생성 ────────────────────────────────────────────────────────
    def _spawn(self, species, initial=False, cell=None, energy=None, hydration=None):
        rng = self.rng
        p = self.params[species]
        if cell is None:
            herb = species not in config.PREY
            # 처음 배치: 초식은 자기 먹이가 있는 칸(풀밭 / 관목)에서 시작한다
            food = self.browse_cells if species in config.BROWSERS else self.meadow_cells
            pool = food if (herb and initial and food) else self.open_cells
            for _ in range(10000):
                cell = pool[rng.randrange(len(pool))]
                if self.occ[cell] is None:
                    break
            else:
                return None
        newborn = energy is not None
        if not newborn:
            lo = 30.0 if self.mode == "train" else 60.0
            energy = rng.uniform(lo, 100.0)
            hydration = rng.uniform(lo, 100.0)
        lifespan = max(2, round(p["max_age"] * rng.uniform(0.8, 1.2)))
        age = 0 if newborn else int(rng.uniform(0, 0.5 * p["max_age"]))
        # 새끼는 maturity(없으면 breed_cooldown) step이 지나야 번식할 수 있다
        cooldown = p.get("maturity", p["breed_cooldown"]) if newborn \
            else rng.randrange(p["breed_cooldown"] + 1)
        a = Animal(self.next_id, species, cell // self.cols, cell % self.cols,
                   energy, hydration, age, lifespan, cooldown)
        self.next_id += 1
        self.occ[cell] = a
        self.animals.append(a)
        self.n_alive[species] += 1
        for prey in a.eats:
            self.hunters[prey].append(a)
        return a

    # ── 조회 API ─────────────────────────────────────────────────────────
    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_water(self, r, c):
        return self.in_bounds(r, c) and self.water[r * self.cols + c]

    def is_meadow(self, r, c):
        return self.in_bounds(r, c) and self.meadow[r * self.cols + c]

    def is_burrow(self, r, c):
        """굴 칸인가(물 칸에서는 항상 False)."""
        return self.in_bounds(r, c) and self.burrow[r * self.cols + c]

    def burrow_count(self):
        """지금 굴 안에 있는 동물 수."""
        occ = self.occ
        return sum(1 for k in self.burrow_cells if occ[k] is not None)

    def grass_level(self, r, c):
        """0.0~1.0. 풀이 자랄 수 없는 칸은 0.0."""
        if not self.in_bounds(r, c):
            return 0.0
        k = r * self.cols + c
        if not self.meadow[k]:
            return 0.0
        return min(1.0, max(0.0, (self.turn - self.eaten[k]) / self.regrow))

    def is_browse(self, r, c):
        """관목(사슴의 먹이)이 자랄 수 있는 칸인가. 풀밭 · 물 · 굴 칸과 겹치지 않는다."""
        return self.in_bounds(r, c) and self.browse[r * self.cols + c]

    def browse_level(self, r, c):
        """0.0~1.0. 관목이 자랄 수 없는 칸은 0.0."""
        if not self.in_bounds(r, c):
            return 0.0
        k = r * self.cols + c
        if not self.browse[k]:
            return 0.0
        return min(1.0, max(0.0, (self.turn - self.browse_eaten[k]) / self.browse_regrow))

    def animal_at(self, r, c):
        return self.occ[r * self.cols + c] if self.in_bounds(r, c) else None

    def can_move(self, a, action):
        """행동이 막히지 않는가(배고픈 포식자가 먹이 칸으로 들어가는 것은 사냥이므로 가능)."""
        if action == 0:
            return True
        dr, dc = config.MOVES[action]
        r, c = a.row + dr, a.col + dc
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        k = r * self.cols + c
        if self.water[k]:
            return False
        if self.burrow[k]:                               # 굴: 쓸 수 있는 종만, 비어 있을 때만
            return a.burrower and self.occ[k] is None
        o = self.occ[k]
        if o is None:
            return True
        return o.species in a.eats and a.energy < self.params[a.species]["hunt_below"]

    def count(self, species):
        return self.n_alive[species]

    def stats(self):
        out = {"turn": self.turn}
        groups = {s: [] for s in config.SPECIES}
        for a in self.animals:
            if a.alive:
                groups[a.species].append(a)
        for s in config.SPECIES:
            g = groups[s]
            out[f"count_{s}"] = len(g)
            out[f"avg_energy_{s}"] = sum(a.energy for a in g) / len(g) if g else 0.0
            out[f"avg_hydration_{s}"] = sum(a.hydration for a in g) / len(g) if g else 0.0
            out[f"births_{s}"] = self.births[s]
            out[f"deaths_{s}"] = self.deaths[s]
        cells = self.meadow_cells
        if cells:
            t, eaten, regrow = self.turn, self.eaten, self.regrow
            total = 0.0
            for k in cells:
                lvl = (t - eaten[k]) / regrow
                total += 1.0 if lvl > 1.0 else lvl
            out["grass_ratio"] = total / len(cells)
        else:
            out["grass_ratio"] = 0.0
        cells = self.browse_cells
        if cells:
            t, eaten, regrow = self.turn, self.browse_eaten, self.browse_regrow
            total = 0.0
            for k in cells:
                lvl = (t - eaten[k]) / regrow
                total += 1.0 if lvl > 1.0 else lvl
            out["browse_ratio"] = total / len(cells)
        else:
            out["browse_ratio"] = 0.0
        for c in CAUSES:
            out[f"deaths_{c}"] = self.cause_total[c]                      # 누적(전 종)
        for s in config.HERBIVORES:
            out[f"deaths_predation_{s}"] = self.metrics[f"deaths_{s}_predation"]   # 누적
        out["hunts"] = self.hunts                                          # 이번 step
        for s in config.CARNIVORES:
            out[f"hunts_{s}"] = self.hunts_by[s]                           # 이번 step
            out[f"immigrants_{s}"] = self.metrics[f"immigrants_{s}"]       # 누적(외부 유입)
        for s in config.SPRINTERS:
            out[f"sprints_{s}"] = self.metrics[f"sprints_{s}"]             # 누적(질주 발동)
        for s in config.RANGED_HUNTERS:
            out[f"far_hunts_{s}"] = self.metrics[f"far_hunts_{s}"]         # 누적(2칸 거리 사냥 성공)
        return out

    # ── 관측 ────────────────────────────────────────────────────────────
    def _build_grass_nb(self, v, k, browse=False):
        """시야 안의 먹이 칸(풀밭, browse=True면 관목) (칸, 방향) 목록. 가까운 순."""
        R, C = self.rows, self.cols
        r, c = divmod(k, C)
        meadow = self.browse if browse else self.meadow
        out = []
        for dy, dx, dist, d in self._offsets[v][k % 8]:
            rr, cc = r + dy, c + dx
            if 0 <= rr < R and 0 <= cc < C and meadow[rr * C + cc]:
                out.append((rr * C + cc, d))
        (self._browse_nb if browse else self._grass_nb)[v][k] = out
        return out

    def _build_prey_nb(self, v, k):
        R, C = self.rows, self.cols
        r, c = divmod(k, C)
        water = self.water
        out = []
        for dy, dx, dist, d in self._offsets[v][k % 8]:
            rr, cc = r + dy, c + dx
            if 0 <= rr < R and 0 <= cc < C and not water[rr * C + cc]:
                out.append((rr * C + cc, d, dist))
        self._prey_nb[v][k] = out
        return out

    def _build_refuge_nb(self, v, k):
        """시야 안의 굴 칸 (칸, 방향, 거리) 목록. 자기 칸이 굴이면 맨 앞에 (k, 0, 0)."""
        R, C = self.rows, self.cols
        r, c = divmod(k, C)
        burrow = self.burrow
        out = [(k, 0, 0)] if burrow[k] else []
        for dy, dx, dist, d in self._offsets[v][k % 8]:
            rr, cc = r + dy, c + dx
            if 0 <= rr < R and 0 <= cc < C and burrow[rr * C + cc]:
                out.append((rr * C + cc, d, dist))
        self._refuge_nb[v][k] = out
        return out

    def water_direction(self, k, a=None):
        """물가 최단 방향(이미 물가면 0). 그 칸에 다른 동물이 있으면 비어 있는 다음 후보를 준다.

        목마른 동물(a)은 목마르지 않은 같은 식성(초식끼리 / 육식끼리)의 동물이 차지한
        물가 칸도 갈 수 있다(자리 양보 규칙).
        """
        d = self.water_dir[k]
        if d:
            occ = self.occ
            dist = self.shore_dist
            side = None
            thirsty = a is not None and a.hydration < 35.0
            burrow = self.burrow
            enter = a is not None and a.burrower
            for cand, cell in self.water_opts[k]:
                if burrow[cell] and not enter:           # 굴은 막힌 칸으로 본다
                    continue
                o = occ[cell]
                if o is None or (thirsty and dist[cell] == 0 and o.carnivore == a.carnivore
                                 and o.hydration >= config.YIELD_ABOVE_HYDRATION):
                    if dist[cell] < dist[k]:
                        return cand                      # 더 가까워지는 빈 칸
                    if side is None:
                        side = [cand]
                    else:
                        side.append(cand)
            if side:                                     # 앞이 막힘: 같은 거리의 빈 칸으로 돌아간다
                # 매번 같은 쪽으로만 비키면 두 칸 사이를 왕복하므로 무작위로 고른다
                return side[0] if len(side) == 1 else side[self.rng.randrange(len(side))]
        return d

    def observe(self, a):
        """이산 상태 튜플을 돌려준다(종 이름이 아니라 역할로 구성한다).

        초식: (energy 단계, hydration 단계, 풀 방향, 물 방향, 위험 방향, 위험 거리)
              '풀 방향' = 그 종의 먹이 방향. 관목을 먹는 종(사슴)은 가장 가까운 먹을 수 있는 관목 칸.
              위험 = 자신을 잡을 수 있는 모든 종 가운데 시야 안에서 가장 가까운 개체
              굴을 쓰는 종(토끼)만, 위협이 보일 때 '풀 방향' 자리에 굴 방향이 들어갈 수 있다.
              질주하는 종(사슴)은 끝에 '달리는 방향'이 붙는다: 위협이 보이고 직전 step에 이동했으면
              그 방향(1~4), 아니면 0. 같은 방향을 이어 가야 질주가 나오므로 그 방향을 알려 준다.
        육식: (energy 단계, hydration 단계, 먹이 방향, 먹이 거리, 물 방향)
              먹이 크기가 여러 가지인 포식자(늑대)는 끝에 먹이 크기(0 없음, 1 소형, 2 대형)가 붙는다.
              같은 거리에 소형과 대형이 함께 있으면 대형을 본다.
              (시야 안의 대형을 항상 우선하게 해 봤더니 늑대가 사슴을 1,000~1,500 step 만에 멸종시켰다.)
        방향 0 = 없음(물은 '이미 물가'), 거리 0 = 없음, 1 = 가까움, 2 = 멂.
        '가까움'의 경계는 종별 config.NEAR_DIST(시야가 넓어져도 구간 수는 2개 그대로).
        """
        k = a.row * self.cols + a.col
        e = a.energy
        h = a.hydration
        eb = 0 if e < 35 else (1 if e < 70 else 2)
        hb = 0 if h < 35 else (1 if h < 70 else 2)
        sp = a.species
        v = self.vision[sp]
        near = self.near[sp]
        occ = self.occ
        if a.carnivore:
            nb = self._prey_nb[v][k] or self._build_prey_nb(v, k)
            eats = a.eats
            # 굴 안의 먹이는 보이지 않는다.
            burrow = self.burrow
            a.target = None
            if a.sizes < 2:
                for cell, d, dist in nb:
                    o = occ[cell]
                    if o is not None and o.species in eats and not burrow[cell]:
                        a.target = o
                        a.target_dist = dist
                        return (eb, hb, d, 1 if dist <= near else 2, self.water_direction(k, a))
                return (eb, hb, 0, 0, self.water_direction(k, a))
            found = None
            fd = fdist = 0
            for cell, d, dist in nb:
                if found is not None and dist > fdist:
                    break
                o = occ[cell]
                if o is not None and o.species in eats and not burrow[cell]:
                    if o.size == 2:                      # 같은 거리면 대형 우선
                        found, fd, fdist = o, d, dist
                        break
                    if found is None:
                        found, fd, fdist = o, d, dist
            wd = self.water_direction(k, a)
            if found is None:
                return (eb, hb, 0, 0, wd, 0)
            a.target = found
            a.target_dist = fdist
            return (eb, hb, fd, 1 if fdist <= near else 2, wd, found.size)
        # 초식: 가장 가까운 먹을 수 있는 먹이(토끼 = 풀, 사슴 = 관목)
        if a.browser:
            nb = self._browse_nb[v][k]
            if nb is None:
                nb = self._build_grass_nb(v, k, True)
            limit = self.turn - self.browse_edible_dt
            eaten = self.browse_eaten
        else:
            nb = self._grass_nb[v][k]
            if nb is None:
                nb = self._build_grass_nb(v, k)
            limit = self.turn - self.edible_dt
            eaten = self.eaten
        g = 0
        for cell, d in nb:
            if eaten[cell] <= limit and occ[cell] is None:   # 다른 동물이 차지한 칸은 제외
                g = d
                break
        wd = self.water_direction(k, a)
        # 자신을 잡을 수 있는 포식자 중 가장 가까운 개체
        r, c = a.row, a.col
        best = v + 1
        threat = None
        for f in self.hunters[sp]:
            dr = f.row - r
            if dr < 0:
                dr = -dr
            if dr < best:
                dc = f.col - c
                if dc < 0:
                    dc = -dc
                if dr + dc < best and f.alive:
                    best = dr + dc
                    threat = f
        a.target = threat
        a.refuge = -1
        if threat is None:
            return (eb, hb, g, wd, 0, 0, 0) if a.sprinter else (eb, hb, g, wd, 0, 0)
        a.target_dist = best
        if a.burrower and (best <= near or eb):
            # 굴을 쓰는 종(토끼)만: 위협이 가깝거나, 위협이 보이는데 굶주리지 않았으면(energy 단계 ≥ 1)
            # 시야 안에 (비어 있는) 굴이 있을 때 '풀 방향' 자리에 굴 방향을 넣는다.
            # 이미 굴 안이면 0(= 여기 머물면 된다). 굶주렸고 위협이 멀면 풀 방향을 그대로 둔다.
            # 굴을 쓰지 않는 종(사슴)의 이 자리는 언제나 풀 방향이다. 상태 수는 그대로 3,375.
            bn = self._refuge_nb[v][k]
            if bn is None:
                bn = self._build_refuge_nb(v, k)
            for cell, d, dist in bn:
                if cell == k or occ[cell] is None:
                    a.refuge = cell
                    a.refuge_dist = dist
                    g = d
                    break
        if a.sprinter:
            return (eb, hb, g, wd, direction(threat.row - r, threat.col - c),
                    1 if best <= near else 2, a.last_dir)
        return (eb, hb, g, wd,
                direction(threat.row - r, threat.col - c), 1 if best <= near else 2)

    def open_path(self, r, c, dr, dc):
        """(r, c)에서 (r + dr, c + dc)까지 물 · 굴이 아닌 칸만 거치는 최단 경로가 있는가.

        2칸 이상 떨어진 먹이를 사냥할 때 쓴다(물이나 굴 너머의 먹이는 잡지 못한다).
        양 끝 칸은 검사하지 않고, 사이 칸에 다른 동물이 있는 것은 상관없다.
        """
        if abs(dr) + abs(dc) <= 1:
            return True
        C = self.cols
        if dr:
            step = 1 if dr > 0 else -1
            k = (r + step) * C + c
            if not self.water[k] and not self.burrow[k] \
                    and self.open_path(r + step, c, dr - step, dc):
                return True
        if dc:
            step = 1 if dc > 0 else -1
            k = r * C + c + step
            if not self.water[k] and not self.burrow[k] \
                    and self.open_path(r, c + step, dr, dc - step):
                return True
        return False

    # ── 진행 ────────────────────────────────────────────────────────────
    def step(self, controllers):
        """모든 개체가 무작위 순서로 한 번씩 행동한다."""
        for s in config.SPECIES:
            self.births[s] = 0
            self.deaths[s] = 0
        self.hunts = 0
        for s in self.hunts_by:
            self.hunts_by[s] = 0
        order = self.animals[:]
        self.rng.shuffle(order)
        observe, apply = self.observe, self._apply
        for a in order:
            if a.alive:
                ctrl = controllers[a.species]
                apply(a, ctrl.act(self, a, observe(a)), controllers)
        self.animals = [a for a in self.animals if a.alive]
        for h in self.hunters:
            self.hunters[h] = [a for a in self.animals if h in a.eats]
        if self.mode == "eval" and config.IMMIGRATION_ENABLED:
            self._immigrate()
        self.turn += 1
        if self.record_history:
            self.history.append(self.stats())

    def _immigrate(self):
        """외부 유입: 수가 아주 적은 포식자 종에 낮은 확률로 가장자리에서 1마리가 들어온다."""
        for s, rule in config.IMMIGRATION.items():
            if self.target_counts.get(s, 0) and self.n_alive[s] < rule["below"]                     and self.rng.random() < rule["prob"]:
                free = [k for k in self.edge_cells if self.occ[k] is None]
                if free:
                    self._spawn(s, cell=free[self.rng.randrange(len(free))])
                    self.metrics["immigrants_" + s] += 1

    def _apply(self, a, action, controllers):
        C = self.cols
        occ = self.occ
        rng = self.rng
        m = self.metrics
        sp = a.species
        carn = a.carnivore
        p = self.params[sp]
        reward = 0.0
        k = a.row * C + a.col
        hungry = carn and a.energy < p["hunt_below"]     # 포만 규칙: 배부르면 사냥하지 않는다
        if hungry:
            m["hungry_steps_" + sp] += 1
        captured = None
        attacked = False
        moved = False
        thirsty = a.hydration < 35.0
        shore0 = self.shore_dist[k]
        swapped = False
        sprint_prob = p.get("sprint_prob", 0.0)
        sprinted = False

        # 1) 이동
        if action:
            dr, dc = config.MOVES[action]
            r, c = a.row + dr, a.col + dc
            if not (0 <= r < self.rows and 0 <= c < C) or self.water[r * C + c] \
                    or (self.burrow[r * C + c] and not a.burrower):
                reward += config.REWARD_BLOCKED          # 경계 · 물 · (들어갈 수 없는) 굴에 막힘
                m["blocked_moves"] += 1
            else:
                t = r * C + c
                o = occ[t]
                if o is None:
                    moved = True
                elif hungry and o.species in a.eats:
                    # 먹이 칸으로 진입 시도 = 덮치기. 실패하면 제자리에 머문다.
                    attacked = True
                    pounce = self._hunt[sp][o.species]["pounce"]
                    if pounce >= 1.0 or rng.random() < pounce:
                        captured = o
                        moved = True
                elif thirsty and self.shore[t] and o.carnivore == carn \
                        and o.hydration >= config.YIELD_ABOVE_HYDRATION:
                    # 자리 양보: 목마른 동물이 물가로 오면, 목마르지 않은 같은 식성 동물은 자리를 바꿔 준다
                    occ[k] = o
                    occ[t] = a
                    o.row, o.col = a.row, a.col
                    a.row, a.col = r, c
                    k = t
                    swapped = True
                if moved:
                    occ[k] = None
                    occ[t] = a
                    a.row, a.col = r, c
                    k = t
                    if captured is not None:
                        self._kill(captured, "predation", controllers)
                    elif sprint_prob and action == a.last_dir \
                            and a.streak >= p["sprint_streak"] and rng.random() < sprint_prob:
                        # 질주: 같은 방향으로 계속 달리던 중이면 한 칸 더 간다(막혀 있으면 1칸만).
                        r2, c2 = r + dr, c + dc
                        if 0 <= r2 < self.rows and 0 <= c2 < C:
                            t2 = r2 * C + c2
                            if occ[t2] is None and not self.water[t2] \
                                    and (a.burrower or not self.burrow[t2]):
                                occ[k] = None
                                occ[t2] = a
                                a.row, a.col = r2, c2
                                k = t2
                                sprinted = True
                                m["sprints_" + sp] += 1
                                if a.target is not None:
                                    m["sprints_threat_" + sp] += 1

        moved = moved or swapped
        if sprint_prob:                                  # 직전 이동 방향 · 연속 횟수 기록
            if moved:
                a.streak = a.streak + 1 if action == a.last_dir else 1
                a.last_dir = action
            else:
                a.streak = 0
                a.last_dir = 0
        if thirsty and moved:                            # 목마를 때 물가 쪽으로 움직였는가
            shore1 = self.shore_dist[k]
            if shore1 < shore0:
                reward += config.SHAPE_WATER
            elif shore1 > shore0:
                reward -= config.SHAPE_WATER

        # 2) 대사 비용
        cost = p["move_cost"] if moved else p["stay_cost"]
        if sprinted:
            cost += p.get("sprint_cost", 0.0)
        if a.energy < config.FAMINE_BELOW:               # 기근 대사: 굶주리면 소모가 준다(종별 배율)
            cost *= p.get("famine_cost_mult", 1.0)
        a.energy -= cost
        a.hydration -= p["thirst"]

        # 3) 물가면 자동으로 마신다
        if self.shore[k]:
            before = a.hydration
            a.hydration = min(config.MAX_HYDRATION, before + config.DRINK_AMOUNT)
            need = 2.0 * (config.MAX_HYDRATION - before) / config.MAX_HYDRATION
            reward += config.REWARD_WATER * (need if need < 1.0 else 1.0)

        # 4) 먹기 / 사냥 + shaping
        if carn:
            # 잡식 먹이(여우): 굶주렸을 때 초원 칸에서 들쥐 · 곤충 · 열매로 조금 연명한다(보상 없음).
            if a.energy < config.FAMINE_BELOW and self.meadow[k]:
                gain = p.get("forage_gain", 0.0)
                if gain:
                    a.energy += gain
                    m["forage_steps_" + sp] += 1
            far = False
            if hungry and not attacked:
                # 사냥 반경(여우 1칸, 늑대 2칸) 안의 먹이 가운데 포획 확률이 가장 높은 것을 고른다
                # (같으면 무작위). 인접은 catch, 2칸 이상은 catch_far로 판정한다.
                near = []
                best = -1.0
                r, c = a.row, a.col
                eats = a.eats
                hunt = self._hunt[sp]
                for dr, dc, dist in self._reach[sp]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.rows and 0 <= cc < C:
                        o = occ[rr * C + cc]
                        if o is not None and o.species in eats \
                                and not self.burrow[rr * C + cc] \
                                and (dist == 1 or self.open_path(r, c, dr, dc)):
                            # 굴 안의 먹이, 물 · 굴 너머(사이 칸이 막힌)의 먹이는 못 잡는다
                            prob = hunt[o.species]["catch" if dist == 1 else "catch_far"]
                            if prob > best:
                                best = prob
                                near = [(o, dist)]
                            elif prob == best:
                                near.append((o, dist))
                if near:                                 # 먹이 하나를 골라 확률 판정
                    o, dist = near[0] if len(near) == 1 else near[rng.randrange(len(near))]
                    if rng.random() < best:
                        captured = o
                        far = dist > 1
                        self._kill(captured, "predation", controllers)
            tgt = a.target
            if hungry and tgt is not None:
                m["chase_steps_" + sp] += 1
            if captured is not None:
                hunt = self._hunt[sp][captured.species]
                before = a.energy
                a.energy = min(config.MAX_ENERGY, before + hunt["energy"])
                a.hydration = min(config.MAX_HYDRATION, a.hydration + hunt["hydration"])
                need = 2.0 * (config.MAX_ENERGY - before) / config.MAX_ENERGY
                reward += hunt["reward"] * (need if need < 1.0 else 1.0)
                self.hunts += 1
                self.hunts_by[sp] += 1
                m["hunts"] += 1
                m["hunts_" + sp] += 1
                m[f"kills_{sp}_{captured.species}"] += 1
                if far:
                    m["far_hunts_" + sp] += 1
                    m[f"far_kills_{sp}_{captured.species}"] += 1
                if tgt is not None:
                    m["close_in_steps_" + sp] += 1
            elif hungry and tgt is not None and tgt.alive:
                d1 = abs(tgt.row - a.row) + abs(tgt.col - a.col)
                if d1 < a.target_dist:
                    reward += config.SHAPE_CHASE
                    m["close_in_steps_" + sp] += 1
                elif d1 > a.target_dist:
                    reward += config.SHAPE_LOSE_PREY
        else:
            # 먹이층: 토끼는 풀밭(meadow), 관목을 먹는 종(사슴)은 관목(browse). 서로의 먹이는 먹지 않는다.
            if a.browser:
                meadow, eaten, regrow = self.browse, self.browse_eaten, self.browse_regrow
                edible = config.BROWSE_EDIBLE_LEVEL
            else:
                meadow, eaten, regrow = self.meadow, self.eaten, self.regrow
                edible = config.GRASS_EDIBLE_LEVEL
            if a.energy < p["eat_below"] and meadow[k]:
                turn = self.turn
                lvl = (turn - eaten[k]) / regrow
                if lvl >= edible:
                    if lvl > 1.0:
                        lvl = 1.0
                    eaten[k] = turn
                    extra = p["graze_extra"]
                    if extra:                            # 큰 초식동물: 인접한 풀도 함께 뜯는다
                        r, c = a.row, a.col
                        for dr, dc in config.MOVES[1:]:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < self.rows and 0 <= cc < C and meadow[rr * C + cc]:
                                t = rr * C + cc
                                lv2 = (turn - eaten[t]) / regrow
                                if lv2 >= edible:
                                    lvl += 1.0 if lv2 > 1.0 else lv2
                                    eaten[t] = turn
                                    extra -= 1
                                    if not extra:
                                        break
                    before = a.energy
                    a.energy = min(config.MAX_ENERGY, before + p["grass_energy"] * lvl)
                    wet = p.get("grass_hydration", 0.0)       # 풀에 든 수분
                    if wet:
                        a.hydration = min(config.MAX_HYDRATION, a.hydration + wet * lvl)
                    need = 2.0 * (config.MAX_ENERGY - before) / config.MAX_ENERGY
                    reward += config.REWARD_GRASS * (need if need < 1.0 else 1.0)
            tgt = a.target
            near_dist = self.near[sp]
            if a.burrower and self.burrow[k]:
                m["shelter_steps_" + sp] += 1
            if tgt is not None and tgt.alive and a.refuge >= 0:
                # 굴이 보였으면: 굴 쪽으로 가거나 굴 안에 머무는 것이 도주다
                # (도주율 통계는 1단계와 같게 위협이 가까울 때만 센다)
                near = a.target_dist <= near_dist
                if near:
                    m["threat_steps_" + sp] += 1
                d1 = abs(a.refuge // C - a.row) + abs(a.refuge % C - a.col)
                if d1 < a.refuge_dist or d1 == 0:
                    reward += config.SHAPE_FLEE
                    m["refuge_steps_" + sp] += 1
                    if near:
                        m["flee_steps_" + sp] += 1
                elif d1 > a.refuge_dist:
                    reward += config.SHAPE_APPROACH_DANGER
                    if near:
                        m["approach_danger_steps_" + sp] += 1
            elif tgt is not None and a.target_dist <= near_dist and tgt.alive:
                d1 = abs(tgt.row - a.row) + abs(tgt.col - a.col)
                m["threat_steps_" + sp] += 1
                if d1 > a.target_dist:
                    reward += config.SHAPE_FLEE
                    m["flee_steps_" + sp] += 1
                elif d1 < a.target_dist:
                    reward += config.SHAPE_APPROACH_DANGER
                    m["approach_danger_steps_" + sp] += 1

        # 5) 사망 판정
        a.age += 1
        a.steps_alive += 1
        m["agent_steps_" + sp] += 1
        cause = None
        if a.energy <= 0:
            cause = "starvation"
        elif a.hydration <= 0:
            cause = "dehydration"
        elif a.age >= a.lifespan:
            cause = "age"
        if cause is not None:
            a.reward_acc += reward
            m["reward_" + sp] += reward
            self._kill(a, cause, controllers)
            return
        reward += config.REWARD_SURVIVE
        a.reward_acc += reward
        m["reward_" + sp] += reward

        # 6) 번식 (eval 모드만)
        if self.mode == "eval":
            if a.cooldown > 0:
                a.cooldown -= 1
            elif a.energy >= p["breed_min_energy"] and a.hydration >= p["breed_min_hydration"] \
                    and rng.random() < p["breed_prob"]:
                self._breed(a, p)

    def _breed(self, a, p):
        n = self.n_alive[a.species]
        if n >= p["population_cap"]:
            return
        # 영역성: 같은 종이 많을수록 번식이 억제된다(territory_limit에 가까워지면 0으로).
        limit = p.get("territory_limit")
        if limit and self.rng.random() >= 1.0 - n / limit:
            return
        C = self.cols
        sites = []
        for dr, dc in config.MOVES[1:]:
            r, c = a.row + dr, a.col + dc
            if 0 <= r < self.rows and 0 <= c < C:
                t = r * C + c
                if not self.water[t] and self.occ[t] is None \
                        and (a.burrower or not self.burrow[t]):
                    sites.append(t)
        if not sites:
            return
        cost = p.get("breed_cost") if config.ENERGY_BUDGET_BREEDING else None
        if cost:
            # 에너지 예산 번식: 부모가 breed_cost를 내고, 새끼는 newborn_energy로 시작한다
            # (차이는 임신 · 수유의 전환 손실). 먹어서 모은 energy만큼만 새끼를 낼 수 있다.
            a.energy -= cost
            child = p.get("newborn_energy", cost)
        else:
            a.energy /= 2.0                               # (예전 방식) 부모 energy를 자식과 반씩
            child = a.energy
        a.cooldown = p["breed_cooldown"]
        self._spawn(a.species, cell=sites[self.rng.randrange(len(sites))],
                    energy=child, hydration=a.hydration)
        self.births[a.species] += 1
        self.metrics["births_" + a.species] += 1

    def _kill(self, a, cause, controllers):
        sp = a.species
        a.alive = False
        a.death_cause = cause
        self.n_alive[sp] -= 1
        cell = a.row * self.cols + a.col
        if self.occ[cell] is a:                           # 포식자가 이미 그 칸에 들어왔을 수 있다
            self.occ[cell] = None
        penalty = config.REWARD_AGE_DEATH if cause == "age" else config.REWARD_DEATH
        a.reward_acc += penalty
        m = self.metrics
        m["reward_" + sp] += penalty
        m["deaths_" + sp] += 1
        m["life_sum_" + sp] += a.steps_alive
        m[f"deaths_{sp}_{cause}"] += 1
        self.deaths[sp] += 1
        self.cause_total[cause] += 1
        controllers[sp].died(self, a)
        if self.mode == "train":
            self._spawn(sp)                               # 개체수 유지: 무작위 위치에 즉시 재생성
