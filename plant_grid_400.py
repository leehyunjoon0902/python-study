"""토끼 공유 SARSA 400회 학습: python plant_grid_400.py --train --watch"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random, SystemRandom


class PlantGrid:
    """grass[row][col]은 그 칸에서 먹을 수 있는 풀의 양이다."""

    def __init__(self, rows=15, cols=20, capacity=10, growth=1, seed=42):
        for value in (rows, cols, capacity, growth):
            if type(value) is not int or value < 1:
                raise ValueError("크기, 최대 풀의 양, 성장량은 양의 정수여야 합니다.")
        self.rows, self.cols = rows, cols
        self.capacity, self.growth = capacity, growth
        self.seed = seed
        self.reset()

    def reset(self):
        rng = Random(self.seed)
        self.turn = 0
        # 중심 좌표는 격자 크기에 대한 비율이며 (행, 열) 순서다.
        centers = [(0.25, 0.25), (0.30, 0.78), (0.76, 0.52)]
        self.meadow = set()
        self.grass = [[0] * self.cols for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                for y, x in centers:
                    dy = (row - int(self.rows * y)) / max(1, self.rows * 0.20)
                    dx = (col - int(self.cols * x)) / max(1, self.cols * 0.17)
                    if dy * dy + dx * dx <= 1:
                        self.meadow.add((row, col))
                        self.grass[row][col] = rng.randint(1, self.capacity)
                        break

    def step(self):
        for row, col in self.meadow:
            self.grass[row][col] = min(self.capacity, self.grass[row][col] + self.growth)
        self.turn += 1

    def consume(self, row, col, amount):
        """남아 있는 풀에서 요청량만큼 차감하고 실제 섭취량을 반환한다."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("격자 밖의 좌표입니다.")
        if type(amount) is not int or amount < 0:
            raise ValueError("섭취량은 0 이상의 정수여야 합니다.")
        eaten = min(self.grass[row][col], amount)
        self.grass[row][col] -= eaten
        return eaten


MOVES = {"정지": (0, 0), "위": (-1, 0), "아래": (1, 0), "왼쪽": (0, -1), "오른쪽": (0, 1)}


@dataclass
class Rabbit:
    id: int
    row: int
    col: int
    energy: int = 10
    age: int = 0
    death_reason: str = ""
    locked: int = 0
    parents: tuple = ()
    next_breeding_turn: int = 0

    @property
    def alive(self):
        return not self.death_reason


class RabbitGrid(PlantGrid):
    def __init__(self, *args, initial_count=10, breeding_wait=2, breeding_cost=2, **kwargs):
        self.initial_count = initial_count
        if any(type(v) is not int or v < 0 for v in (breeding_wait, breeding_cost)):
            raise ValueError("번식 대기시간과 비용은 0 이상의 정수여야 합니다.")
        self.breeding_wait = breeding_wait
        self.breeding_cost = breeding_cost
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.rng = Random(self.seed)
        slots = self.rows * self.cols * 2
        if not 1 <= self.initial_count <= slots:
            raise ValueError("초기 개체 수가 격자 수용 범위를 벗어났습니다.")
        self.rabbits = []
        for index, slot in enumerate(self.rng.sample(range(slots), self.initial_count), 1):
            row, col = divmod(slot // 2, self.cols)
            self.rabbits.append(Rabbit(index, row, col))
        self.next_id = self.initial_count + 1
        self.births = self.deaths = self.last_eaten = 0

    def observe(self, rabbit=None):
        if rabbit is None:
            rabbit = self.rabbits[0]
        food = {}
        for action, (dr, dc) in MOVES.items():
            r, c = rabbit.row + dr, rabbit.col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                food[action] = self.grass[r][c]
        return {"energy": rabbit.energy, "food": food}

    def step(self, action=None, *, actions=None):
        if not self.rabbits:
            raise RuntimeError("모든 토끼가 사망했습니다.")
        leader = self.rabbits[0]
        if action is not None and action not in self.observe(leader)["food"]:
            raise ValueError("선택할 수 없는 방향입니다.")
        order = self.rabbits.copy()
        self.rng.shuffle(order)
        occupancy = Counter((r.row, r.col) for r in order)
        groups = defaultdict(list)
        self.last_eaten = self.births = self.deaths = 0
        self.rewards = Counter()
        self.matured = 0
        # 무작위 순서로 빈자리를 배정하고 같은 순서로 먹이를 나눈다.
        for rabbit in order:
            choice = "정지"
            if not rabbit.locked:
                if actions is not None:
                    choice = actions[rabbit.id]
                elif rabbit is leader and action is not None:
                    choice = action
                else:
                    choice = self.rng.choice(list(self.observe(rabbit)["food"]))
            dr, dc = MOVES[choice]
            target = rabbit.row + dr, rabbit.col + dc
            if occupancy[target] >= 2:
                choice = "정지"
            else:
                occupancy[rabbit.row, rabbit.col] -= 1
                rabbit.row, rabbit.col = target
                occupancy[target] += 1
            self.feed_and_age(rabbit, choice)
            if rabbit.alive:
                groups[rabbit.row, rabbit.col].append(rabbit)
                if rabbit.parents and rabbit.age == 3:
                    self.matured += 1
                    for parent in rabbit.parents:
                        self.rewards[parent] += 2
            else:
                if rabbit.death_reason == "굶주림":
                    self.rewards[rabbit.id] -= 3
                occupancy[rabbit.row, rabbit.col] -= 1
                self.deaths += 1
        self.rabbits = [r for r in self.rabbits if r.alive]
        for (row, col), parents in groups.items():
            if len(parents) != 2:
                continue
            turn = self.turn + 1
            if any(p.next_breeding_turn > turn or p.energy <= self.breeding_cost for p in parents):
                continue
            sites = [(row + dr, col + dc) for dr, dc in MOVES.values()
                     if (dr or dc) and 0 <= row + dr < self.rows
                     and 0 <= col + dc < self.cols and occupancy[row + dr, col + dc] < 2]
            if sites:
                site = self.rng.choice(sites)
                ids = tuple(r.id for r in parents)
                self.rabbits.append(Rabbit(self.next_id, *site, locked=3, parents=ids))
                for parent in parents:
                    parent.energy -= self.breeding_cost
                    parent.next_breeding_turn = turn + self.breeding_wait + 1
                    self.rewards[parent.id] += 1
                occupancy[site] += 1
                self.next_id += 1
                self.births += 1
        super().step()

    def feed_and_age(self, rabbit, action):
        rabbit.locked = max(0, rabbit.locked - 1)
        rabbit.energy -= 1 if action == "정지" else 2
        eaten = self.consume(rabbit.row, rabbit.col, min(3, 20 - rabbit.energy))
        self.last_eaten += eaten
        rabbit.energy += eaten
        rabbit.age += 1
        if rabbit.age >= 30:
            rabbit.death_reason = "수명 종료"
        elif rabbit.energy <= 0:
            rabbit.death_reason = "굶주림"
        rabbit.energy = max(0, rabbit.energy)


class SharedSARSA:
    def __init__(self, vision_radius=1):
        if type(vision_radius) is not int or vision_radius < 1:
            raise ValueError("시야 반경은 1 이상의 정수여야 합니다.")
        self.vision_radius = vision_radius
        self.q = {}

    def visible_directions(self, env, rabbit, counts, food):
        # 방향별로 한 걸음 다가갈 수 있는 시야 내 풀과 짝 후보를 요약한다.
        scores = {a: (v, 0 if a == "정지" else -1) for a, v in food.items()}
        partners = set()
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            width = self.vision_radius - abs(dy)
            for dx in range(-width, width + 1):
                distance = abs(dy) + abs(dx)
                row, col = rabbit.row + dy, rabbit.col + dx
                if not distance or not (0 <= row < env.rows and 0 <= col < env.cols):
                    continue
                for action, (dr, dc) in MOVES.items():
                    if dy * dr + dx * dc <= 0:
                        continue
                    scores[action] = max(scores[action], (env.grass[row][col], -distance))
                    if counts[row, col] == 1:
                        partners.add(action)
        richest = max(scores, key=scores.get)
        return (richest if scores[richest][0] else None,
                tuple(a for a in MOVES if a in partners))

    def choose(self, env, rng, epsilon):
        counts = Counter((r.row, r.col) for r in env.rabbits)
        decisions = {}
        for r in env.rabbits:
            food = env.observe(r)["food"]
            richest, neighbors = self.visible_directions(env, r, counts, food)
            state = (min(r.energy // 7, 2), r.age // 10, bool(food["정지"]),
                     richest, counts[r.row, r.col] == 2, tuple(neighbors), bool(r.locked),
                     max(0, r.next_breeding_turn - env.turn - 1))
            legal = ["정지"] if r.locked else list(food)
            if rng.random() < epsilon:
                chosen = rng.choice(legal)
            else:
                scores = [self.q.get((state, a), 0.0) for a in legal]
                best = max(scores)
                chosen = rng.choice([a for a, v in zip(legal, scores) if v == best])
            decisions[r.id] = (state, chosen)
        return decisions


def episode(seed, policy=None, learn=False, horizon=300):
    env = RabbitGrid(seed=seed)
    rng = Random(seed + 100000)
    epsilon = 0.15 if learn else 0
    decisions = policy.choose(env, rng, epsilon) if policy else {}
    eligible_births = matured = 0
    while env.rabbits and env.turn < horizon:
        actions = {i: pair[1] for i, pair in decisions.items()} if policy else None
        env.step(actions=actions)
        if env.turn <= horizon - 3:
            eligible_births += env.births
        matured += env.matured
        if policy:
            following = policy.choose(env, rng, epsilon)
            if learn:
                # 각 개체의 실제 다음 행동으로 SARSA 목표를 계산한 후 공유 표에 반영한다.
                targets = []
                for i, pair in decisions.items():
                    future = policy.q.get(following[i], 0.0) if i in following else 0.0
                    targets.append((pair, env.rewards[i] + 0.95 * future))
                for pair, target in targets:
                    old = policy.q.get(pair, 0.0)
                    policy.q[pair] = old + 0.1 * (target - old)
            decisions = following
    return bool(env.rabbits), env.turn, eligible_births, matured


def experiment():
    from statistics import mean
    seeds = range(1000, 1010)
    def report(label, policy):
        runs = [episode(seed, policy) for seed in seeds]
        deaths = [turn for alive, turn, _, _ in runs if not alive]
        births = sum(run[2] for run in runs)
        print(label, "survived:", sum(run[0] for run in runs), "/", len(runs),
              "mean extinction turn:", mean(deaths) if deaths else None,
              "3-turn survivors / eligible births:", sum(run[3] for run in runs), "/", births,
              flush=True)
    report("Random", None)
    policy = SharedSARSA()
    for seed in range(400):
        alive, turns, _, _ = episode(seed, policy, learn=True, horizon=60)
        print("Training", seed + 1, "turns:", turns, "Q entries:", len(policy.q), flush=True)
    report("SARSA", policy)
    return policy


def show(env, policy=None):
    import tkinter as tk

    root = tk.Tk()
    root.title("먹이사슬 실험 · 식물 성장")
    size = 25
    canvas = tk.Canvas(root, width=env.cols * size, height=env.rows * size,
                       highlightthickness=0)
    canvas.pack()
    status = tk.Label(root)
    status.pack()
    timer = None
    has_rabbit = isinstance(env, RabbitGrid)
    controls_by_action = {}
    policy_rng = Random(env.seed + 100000 if env.seed is not None else None)
    if has_rabbit:
        mode = "학습 정책" if policy else "무작위 정책"
        root.title(f"먹이사슬 실험 · {mode}")

    def draw():
        canvas.delete("all")
        for r, row in enumerate(env.grass):
            for c, amount in enumerate(row):
                fraction = amount / env.capacity
                rgb = [round(a + (b - a) * fraction)
                       for a, b in zip((242, 231, 203), (28, 107, 57))]
                canvas.create_rectangle(c*size, r*size, (c+1)*size, (r+1)*size,
                                        fill=("#%02x%02x%02x" % tuple(rgb)
                                              if (r, c) in env.meadow else "#baaba0"), outline="white")
        total = sum(map(sum, env.grass))
        if has_rabbit:
            cells = defaultdict(list)
            for rabbit in env.rabbits:
                cells[rabbit.row, rabbit.col].append(rabbit)
            for (row, col), animals in cells.items():
                for index, rabbit in enumerate(animals):
                    x = col * size + (12 if len(animals) == 1 else 6 + 12 * index)
                    y = row * size + 13
                    color = "#ffc766" if rabbit.locked else "white"
                    canvas.create_oval(x-4, y-10, x-1, y, fill=color)
                    canvas.create_oval(x+1, y-10, x+4, y, fill=color)
                    canvas.create_oval(x-5, y-3, x+5, y+7, fill=color)
            for action, button in controls_by_action.items():
                enabled = bool(env.rabbits) and action in env.observe()["food"]
                button.config(state="normal" if enabled else "disabled")
            play.config(state="normal" if env.rabbits else "disabled")
            status.config(text=f"{env.turn}턴  |  토끼 {len(env.rabbits)}마리  |  이번 턴 출생 {env.births} · 사망 {env.deaths}"
                               f"\n누적 출생 {env.next_id - env.initial_count - 1}  |  배치 {env.seed}  |  주황색은 이동 대기")
            return
        status.config(text=f"{env.turn}턴   |   전체 풀의 양 {total}   |   칸을 클릭하면 풀을 먹습니다")

    def advance(action=None):
        if has_rabbit:
            if not env.rabbits:
                return
            food = env.observe()["food"]
            if policy and action is None:
                decisions = policy.choose(env, policy_rng, 0)
                env.step(actions={i: pair[1] for i, pair in decisions.items()})
                if not env.rabbits:
                    stop()
                draw()
                return
            if action is None:
                action = env.rng.choice(list(food))
            if action not in food:
                return
            env.step(action)
            if not env.rabbits:
                stop()
        else:
            env.step()
        draw()

    def tick():
        nonlocal timer
        advance()
        if not has_rabbit or env.rabbits:
            timer = root.after(500, tick)

    def stop():
        nonlocal timer
        if timer is not None:
            root.after_cancel(timer)
            timer = None
        play.config(text="자동 진행")

    def toggle():
        if timer is None:
            play.config(text="일시정지")
            tick()
        else:
            stop()

    def reset():
        stop()
        if has_rabbit:
            env.seed = SystemRandom().randrange(2**32)
            policy_rng.seed(env.seed + 100000)
        env.reset()
        draw()

    def harvest(event):
        row, col = event.y // size, event.x // size
        if 0 <= row < env.rows and 0 <= col < env.cols:
            env.consume(row, col, env.capacity)
            draw()

    controls = tk.Frame(root)
    controls.pack()
    if has_rabbit and policy is None:
        for action, key in zip(MOVES, ("space", "Up", "Down", "Left", "Right")):
            button = tk.Button(controls, text=action, command=lambda a=action: advance(a))
            button.pack(side="left")
            controls_by_action[action] = button
            root.bind(f"<{key}>", lambda event, a=action: advance(a))
    else:
        tk.Button(controls, text="한 턴 진행", command=advance).pack(side="left")
    play = tk.Button(controls, text="자동 진행", command=toggle)
    play.pack(side="left")
    tk.Button(controls, text="새 배치" if has_rabbit else "초기화", command=reset).pack(side="left")
    canvas.bind("<Button-1>", harvest)
    draw()
    root.mainloop()


if __name__ == "__main__":
    import sys
    if "--random" in sys.argv:
        show(RabbitGrid(seed=SystemRandom().randrange(2**32)))
    else:
        trained = experiment()
        if "--train" not in sys.argv or "--watch" in sys.argv:
            show(RabbitGrid(seed=SystemRandom().randrange(2**32)), trained)
