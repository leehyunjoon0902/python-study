"""food_chain 코어 테스트.

실행: python -m unittest tests.test_food_chain_core -v   (pytest로도 실행 가능)
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from food_chain import config                                    # noqa: E402
from food_chain.rl import (ENCODERS, QPolicy, QTable, RandomPolicy, RulePolicy,  # noqa: E402
                           STATE_SHAPE, decode, encode_fox, encode_rabbit, encode_wolf,
                           n_states)
from food_chain.simulation import (NO_PREDATORS, Simulation, make_controllers,  # noqa: E402
                                   run_population, stats_keys)
from food_chain.train import LOG_COLUMNS, plan, train            # noqa: E402
from food_chain.world import World, direction, level_bin         # noqa: E402


class Scripted:
    """개체 id별로 정해진 행동을 내는 테스트용 정책."""

    def __init__(self, actions=None, default=0):
        self.actions = actions or {}
        self.default = default
        self.dead = []

    def act(self, world, a, obs):
        return self.actions.get(a.id, self.default)

    def died(self, world, a):
        self.dead.append(a)


def empty_world(seed=3, mode="eval", cover=False):
    """동물이 없는 세계. cover=False면 굴도 없앤다(기본 규칙 테스트용)."""
    w = World(seed=seed, mode=mode, randomize=False)
    w.clear_animals()
    if not cover:
        n = w.rows * w.cols
        w.burrow = [False] * n
        w.burrow_cells = []
    return w


def burrow_cell(w):
    """굴 칸 가운데 왼쪽 두 칸이 평범한 땅(굴 · 물 · 물가 아님)인 것."""
    def ok(r, c):
        k = r * w.cols + c
        return (c >= 2 and w.burrow[k] and not w.burrow[k - 1] and not w.burrow[k - 2]
                and not w.water[k - 1] and not w.water[k - 2]
                and not w.shore[k - 1] and not w.shore[k - 2] and not w.shore[k])
    return find_cell(w, ok)


def wide_cell(w, margin=4):
    """주변 (2·margin+1)² 칸이 모두 땅이고 물가도 아닌 칸."""
    def ok(r, c):
        if not (margin <= r < w.rows - margin and margin <= c < w.cols - margin):
            return False
        return all(not w.water[(r + dr) * w.cols + c + dc]
                   and not w.shore[(r + dr) * w.cols + c + dc]
                   for dr in range(-margin, margin + 1) for dc in range(-margin, margin + 1))
    return find_cell(w, ok)


def ctrls(default=None, **by_species):
    """종별 controller dict. 지정하지 않은 종은 정지만 하는 Scripted."""
    return {s: by_species.get(s) or default or Scripted() for s in config.SPECIES}


FOX_RABBIT = config.HUNT[("fox", "rabbit")]
WOLF_RABBIT = config.HUNT[("wolf", "rabbit")]
WOLF_DEER = config.HUNT[("wolf", "deer")]


class patched:
    """with patched(dict, key=value): 블록 안에서만 dict 값을 바꾼다."""

    def __init__(self, target, **values):
        self.target, self.values = target, values

    def __enter__(self):
        self.old = {k: self.target[k] for k in self.values}
        self.target.update(self.values)

    def __exit__(self, *exc):
        self.target.update(self.old)


def food_layer(w, species):
    """그 종이 먹는 먹이층의 (칸 여부 목록 이름, 뜯긴 시각 목록, 재생 시간)."""
    if species in config.BROWSERS:
        return "browse", w.browse_eaten, w.browse_regrow
    return "meadow", w.eaten, w.regrow


def only_food_at(w, species, cells):
    """species의 먹이가 cells에만(다 자란 상태로) 있게 한다. 관측용 이웃 목록도 비운다."""
    n = w.rows * w.cols
    name, eaten, regrow = food_layer(w, species)
    mask = [False] * n
    for r, c in cells:
        mask[r * w.cols + c] = True
        eaten[r * w.cols + c] = -10 * regrow
    setattr(w, name, mask)
    w._grass_nb = {v: [None] * n for v in w._grass_nb}
    w._browse_nb = {v: [None] * n for v in w._browse_nb}


def food_level(w, species, r, c):
    return w.browse_level(r, c) if species in config.BROWSERS else w.grass_level(r, c)


def put(w, species, r, c, energy=80.0, hydration=80.0):
    return w._spawn(species, cell=r * w.cols + c, energy=energy, hydration=hydration)


def find_cell(w, ok):
    for r in range(w.rows):
        for c in range(w.cols):
            if ok(r, c):
                return r, c
    raise AssertionError("조건에 맞는 칸이 없습니다.")


def open_cell(w, need=lambda w, r, c: True):
    """상하좌우가 모두 땅이고 물가가 아닌 칸."""
    def ok(r, c):
        if not (1 <= r < w.rows - 1 and 1 <= c < w.cols - 2):
            return False
        cells = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1), (r, c + 2)]
        return all(not w.is_water(*x) and not w.shore[x[0] * w.cols + x[1]] for x in cells) \
            and need(w, r, c)
    return find_cell(w, ok)


class ContractTest(unittest.TestCase):
    def test_config_contract(self):
        self.assertEqual(config.SPECIES, ("rabbit", "deer", "fox", "wolf"))
        self.assertEqual(config.HERBIVORES, ("rabbit", "deer"))
        self.assertEqual(config.CARNIVORES, ("fox", "wolf"))
        self.assertEqual(config.SPECIES_LABEL,
                         {"rabbit": "토끼", "deer": "사슴", "fox": "여우", "wolf": "늑대"})
        self.assertEqual(config.PREY, {"fox": ("rabbit",), "wolf": ("deer", "rabbit")})
        self.assertEqual(config.PREDATORS, {"rabbit": ("fox", "wolf"), "deer": ("wolf",)})
        self.assertEqual((config.ROWS, config.COLS), (50, 75))
        self.assertEqual(config.MAX_ENERGY, 100)
        self.assertEqual(config.MAX_HYDRATION, 100)
        self.assertEqual(config.ALGORITHM, {"rabbit": "sarsa", "deer": "sarsa",
                                            "fox": "qlearning", "wolf": "qlearning"})
        self.assertEqual(set(config.PARAMS), set(config.SPECIES))
        self.assertGreaterEqual(config.EVAL_STEPS, 4000)

    def test_carnivores_see_farther_than_herbivores(self):
        self.assertEqual(set(config.VISION), set(config.SPECIES))
        for c in config.CARNIVORES:                      # 포식–피식 쌍 기준
            for h in config.PREY[c]:
                self.assertGreaterEqual(config.VISION[c], config.VISION[h] + 1)
        self.assertEqual((config.VISION["rabbit"], config.VISION["deer"]), (4, 5))
        for s in config.SPECIES:                         # 거리 2구간: 가까움 경계는 시야 안쪽
            self.assertLess(config.NEAR_DIST[s], config.VISION[s])
        self.assertEqual(config.VISION["wolf"], max(config.VISION.values()))   # 늑대가 가장 넓다
        self.assertGreater(config.VISION["wolf"], config.VISION["fox"])

    def test_species_ratios_follow_real_world_direction(self):
        """수치의 방향: 사슴은 토끼보다 오래 살고 느리게 번식, 포식자는 먹이보다 적고, 늑대 < 여우."""
        P = config.PARAMS
        self.assertGreater(P["deer"]["max_age"], P["rabbit"]["max_age"])
        self.assertGreater(P["deer"]["breed_cooldown"], P["rabbit"]["breed_cooldown"])
        self.assertGreater(P["wolf"]["max_age"], P["fox"]["max_age"])
        self.assertLess(P["wolf"]["initial_count"], P["fox"]["initial_count"])
        self.assertLess(P["fox"]["initial_count"], P["deer"]["initial_count"])
        self.assertLess(P["deer"]["initial_count"], P["rabbit"]["initial_count"])
        self.assertGreater(config.HUNT[("wolf", "deer")]["energy"],
                           config.HUNT[("wolf", "rabbit")]["energy"])       # 큰 먹이 = 많은 energy
        # 토끼는 여우의 먹이다: 여우→토끼 포획 확률이 늑대→토끼보다 높다.
        # (여우→토끼의 step당 확률은 늑대→사슴보다 낮아졌다. 토끼는 여우를 떨쳐 내지 못해 추격이
        #  여러 step 이어지므로 추격 한 번의 성공률은 여전히 높다. 높게 두면 토끼가 바닥나 함께 멸종했다.)
        for key in ("pounce", "catch"):
            self.assertGreater(config.HUNT[("fox", "rabbit")][key],
                               config.HUNT[("wolf", "rabbit")][key])
        # 여우는 배고플 때만 사냥하고(포만 규칙), 번식은 사냥으로 energy를 채워야만 가능하다
        self.assertLess(P["fox"]["hunt_below"], P["fox"]["breed_min_energy"])
        self.assertGreaterEqual(P["fox"]["hunt_below"] + config.HUNT[("fox", "rabbit")]["energy"],
                                P["fox"]["breed_min_energy"])

    def test_metabolism_follows_body_mass(self):
        """체중당 대사율: 토끼 > 여우 > 사슴 > 늑대(작은 동물일수록 빨리 소모). 정지는 이동의 절반."""
        P = config.PARAMS
        cost = [P[s]["move_cost"] for s in ("rabbit", "fox", "deer", "wolf")]
        self.assertEqual(cost, sorted(cost, reverse=True))
        self.assertEqual(len(set(cost)), 4)
        self.assertEqual(cost, [1.0, 0.8, 0.6, 0.5])
        for s in config.SPECIES:
            self.assertAlmostEqual(P[s]["stay_cost"], P[s]["move_cost"] / 2)
            # 탈수가 아사보다 빠르다: 물 없이 버티는 시간 ≤ 계속 움직이며 굶는 시간 < 쉬면서 굶는 시간
            self.assertGreaterEqual(P[s]["thirst"], P[s]["move_cost"])
            self.assertGreater(P[s]["thirst"], P[s]["stay_cost"])
        for s in config.HERBIVORES:                      # 먹이에 든 수분
            self.assertGreater(P[s]["grass_hydration"], 0.0)
            self.assertLess(P[s]["grass_hydration"], P[s]["grass_energy"])
        for hunt in config.HUNT.values():
            self.assertGreater(hunt["hydration"], 0.0)
        # 기근 대사 절감: 늑대는 현실 범위(10~30%)에 가깝게
        self.assertGreaterEqual(P["wolf"]["famine_cost_mult"], 0.7)

    def test_grass_gives_a_little_water(self):
        for species in config.HERBIVORES:
            w = empty_world()
            r, c = open_cell(w)
            only_food_at(w, species, [(r, c)])           # 이 칸의 먹이만 남긴다(사슴의 인접 칸 뜯기 제외)
            a = put(w, species, r, c, energy=30.0, hydration=50.0)
            w.step(ctrls())
            p = w.params[species]
            self.assertAlmostEqual(a.hydration, 50.0 - p["thirst"] + p["grass_hydration"])
            self.assertAlmostEqual(a.energy, 30.0 - p["stay_cost"] + p["grass_energy"])
            w.step(ctrls())                              # 풀이 없으면 수분도 없다
            self.assertAlmostEqual(a.hydration, 50.0 - 2 * p["thirst"] + p["grass_hydration"])

    def test_stats_keys(self):
        expected = {"turn", "grass_ratio", "browse_ratio", "hunts",
                    "deaths_starvation", "deaths_dehydration",
                    "deaths_predation", "deaths_age"}
        for s in ("rabbit", "deer", "fox", "wolf"):
            expected |= {f"count_{s}", f"avg_energy_{s}", f"avg_hydration_{s}",
                         f"births_{s}", f"deaths_{s}"}
        expected |= {"deaths_predation_rabbit", "deaths_predation_deer",   # 누적(피식 종별)
                     "hunts_fox", "hunts_wolf",                             # 이번 step(포식 종별)
                     "immigrants_fox", "immigrants_wolf",                   # 누적(외부 유입)
                     "sprints_deer", "far_hunts_wolf"}       # 누적(질주 발동 / 2칸 거리 사냥)
        sim = Simulation(seed=1, policy_mode="rule")
        self.assertEqual(set(sim.world.stats()), expected)
        self.assertEqual(set(stats_keys()), expected)
        self.assertEqual(len(sim.world.history), 1)          # reset 직후 0번째
        sim.step()
        self.assertEqual(len(sim.world.history), 2)
        self.assertEqual(set(sim.world.history[-1]), expected)
        self.assertEqual(sim.world.history[-1]["turn"], 1)
        self.assertEqual(sim.world.turn, 1)

    def test_world_and_animal_api(self):
        sim = Simulation(seed=5, policy_mode="random")
        w = sim.world
        self.assertEqual((w.rows, w.cols), (50, 75))
        self.assertEqual(sim.policy_mode, "random")
        a = w.animals[0]
        for name in ("id", "species", "row", "col", "energy", "hydration", "age", "alive"):
            self.assertTrue(hasattr(a, name))
        self.assertIn(a.species, config.SPECIES)
        levels = [w.grass_level(r, c) for r in range(w.rows) for c in range(w.cols)]
        self.assertTrue(all(0.0 <= x <= 1.0 for x in levels))
        for r in range(w.rows):
            for c in range(w.cols):
                if not w.is_meadow(r, c):
                    self.assertEqual(w.grass_level(r, c), 0.0)
                self.assertFalse(w.is_water(r, c) and w.is_meadow(r, c))
        for s in config.SPECIES:
            self.assertGreater(w.stats()[f"count_{s}"], 0)
            self.assertGreater(w.stats()[f"avg_energy_{s}"], 0.0)
            self.assertEqual(w.count(s), w.stats()[f"count_{s}"])
        self.assertTrue(0.0 < w.stats()["grass_ratio"] <= 1.0)

    def test_average_is_zero_when_species_absent(self):
        w = empty_world()
        st = w.stats()
        self.assertEqual(st["count_fox"], 0)
        self.assertEqual(st["avg_energy_fox"], 0.0)
        self.assertEqual(st["avg_hydration_rabbit"], 0.0)

    def test_unknown_policy_and_missing_tables(self):
        with self.assertRaises(ValueError):
            Simulation(seed=1, policy_mode="nope")
        old = config.Q_TABLE_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                config.Q_TABLE_DIR = Path(d)
                with self.assertRaises(FileNotFoundError):
                    Simulation(seed=1, policy_mode="trained")
        finally:
            config.Q_TABLE_DIR = old

    def test_train_log_columns(self):
        self.assertEqual(LOG_COLUMNS[:2], ["episode", "epsilon"])
        self.assertIn("hunts", LOG_COLUMNS)
        for s in config.SPECIES:
            for name in ("mean_reward", "mean_lifespan", "q_entries"):
                self.assertIn(f"{name}_{s}", LOG_COLUMNS)
        self.assertEqual(len(LOG_COLUMNS), len(set(LOG_COLUMNS)))

    def test_shore_cells_are_plentiful(self):
        """물웅덩이를 키웠다: 물가 칸이 초기 개체수의 절반 이상."""
        total = sum(config.PARAMS[s]["initial_count"] for s in config.SPECIES)
        for seed in range(5):
            w = World(seed=seed)
            self.assertGreaterEqual(sum(w.shore), 150)
            self.assertGreaterEqual(sum(w.shore), total / 2)


class TerrainTest(unittest.TestCase):
    def test_water_is_reachable_everywhere(self):
        for seed in range(8):
            w = World(seed=seed)
            far = max(w.shore_dist[k] for k in w.land_cells)
            self.assertLessEqual(far, config.WATER_MAX_DIST)
            self.assertGreater(sum(w.water), 0)

    def test_water_direction_leads_to_shore(self):
        w = World(seed=2)
        for k in w.land_cells[::7]:
            r, c = divmod(k, w.cols)
            for _ in range(config.WATER_MAX_DIST + 1):
                d = w.water_dir[r * w.cols + c]
                if d == 0:
                    break
                r, c = r + config.MOVES[d][0], c + config.MOVES[d][1]
                self.assertFalse(w.is_water(r, c))
            self.assertTrue(w.shore[r * w.cols + c])

    def test_direction_and_bins(self):
        self.assertEqual(direction(0, 0), 0)
        self.assertEqual(direction(-2, 1), 1)
        self.assertEqual(direction(2, -2), 2)      # |dy| == |dx|이면 세로 우선
        self.assertEqual(direction(1, -3), 3)
        self.assertEqual(direction(0, 4), 4)
        self.assertEqual([level_bin(v) for v in (0, 34.9, 35, 69.9, 70, 100)],
                         [0, 0, 1, 1, 2, 2])


class RulesTest(unittest.TestCase):
    def test_same_seed_reproduces(self):
        runs = []
        for _ in range(2):
            sim = Simulation(seed=11, policy_mode="rule")
            for _ in range(120):
                sim.step()
            runs.append((sim.world.history,
                         [(a.id, a.species, a.row, a.col, a.energy, a.hydration, a.age)
                          for a in sim.world.animals]))
        self.assertEqual(runs[0], runs[1])
        other = Simulation(seed=12, policy_mode="rule")
        for _ in range(120):
            other.step()
        self.assertNotEqual(runs[0][0], other.world.history)

    def test_reset_with_seed_reproduces(self):
        sim = Simulation(seed=4, policy_mode="random")
        first = [(a.species, a.row, a.col) for a in sim.world.animals]
        for _ in range(30):
            sim.step()
        sim.reset(seed=4)
        self.assertEqual(first, [(a.species, a.row, a.col) for a in sim.world.animals])
        self.assertEqual(sim.world.turn, 0)
        self.assertEqual(len(sim.world.history), 1)
        sim.reset()                                  # seed=None → 새 무작위 배치
        self.assertEqual(sim.world.turn, 0)

    def test_drinking_next_to_water(self):
        w = empty_world()
        r, c = find_cell(w, lambda r, c: w.shore[r * w.cols + c])
        rabbit = put(w, "rabbit", r, c, energy=95.0, hydration=40.0)   # 배불러 풀은 안 뜯는다
        w.step(ctrls())
        thirst = w.params["rabbit"]["thirst"]
        self.assertAlmostEqual(rabbit.hydration, 40.0 - thirst + config.DRINK_AMOUNT)
        rabbit.hydration = 95.0
        w.step(ctrls())
        self.assertEqual(rabbit.hydration, config.MAX_HYDRATION)     # 상한 100

    def test_no_drinking_away_from_water(self):
        w = empty_world()
        r, c = open_cell(w)
        rabbit = put(w, "rabbit", r, c, hydration=40.0)
        w.step(ctrls())
        self.assertAlmostEqual(rabbit.hydration, 40.0 - w.params["rabbit"]["thirst"])

    def test_water_and_border_block_movement(self):
        w = empty_world()
        r, c = find_cell(w, lambda r, c: not w.is_water(r, c) and w.is_water(r, c + 1))
        rabbit = put(w, "rabbit", r, c)
        w.step(ctrls(rabbit=Scripted(default=4)))
        self.assertEqual((rabbit.row, rabbit.col), (r, c))
        self.assertLess(rabbit.reward_acc, config.REWARD_SURVIVE + config.REWARD_WATER)
        self.assertEqual(w.metrics["blocked_moves"], 1)
        corner = put(w, "rabbit", 0, 0) if not w.is_water(0, 0) else None
        if corner:
            w.step(ctrls(rabbit=Scripted({corner.id: 1})))
            self.assertEqual((corner.row, corner.col), (0, 0))

    def test_thirsty_animal_takes_over_shore_cell(self):
        """자리 양보: 목마른 토끼가 물가로 오면 목마르지 않은 토끼와 자리를 바꾼다."""
        w = empty_world()

        def ok(r, c):
            return (c + 1 < w.cols and w.shore[r * w.cols + c] and not w.is_water(r, c + 1)
                    and not w.shore[r * w.cols + c + 1])
        r, c = find_cell(w, ok)
        camper = put(w, "rabbit", r, c, hydration=95.0)
        thirsty = put(w, "rabbit", r, c + 1, hydration=20.0)
        self.assertNotEqual(w.observe(thirsty)[3], 0)
        w._apply(thirsty, 3, ctrls())
        self.assertEqual((thirsty.row, thirsty.col), (r, c))
        self.assertEqual((camper.row, camper.col), (r, c + 1))
        self.assertIs(w.animal_at(r, c), thirsty)
        self.assertIs(w.animal_at(r, c + 1), camper)
        self.assertAlmostEqual(thirsty.hydration,
                               20.0 - w.params["rabbit"]["thirst"] + config.DRINK_AMOUNT)
        # 물가의 동물도 목마르면 양보하지 않는다
        w2 = empty_world()
        a = put(w2, "rabbit", r, c, hydration=50.0)
        b = put(w2, "rabbit", r, c + 1, hydration=20.0)
        w2._apply(b, 3, ctrls())
        self.assertEqual((a.row, a.col, b.row, b.col), (r, c, r, c + 1))

    def test_starvation(self):
        w = empty_world()
        r, c = open_cell(w, lambda w, r, c: not w.is_meadow(r, c))
        rabbit = put(w, "rabbit", r, c, energy=0.2)
        ctrl = Scripted()
        w.step(ctrls(rabbit=ctrl))
        self.assertFalse(rabbit.alive)
        self.assertEqual(rabbit.death_cause, "starvation")
        self.assertEqual(w.animals, [])
        self.assertEqual(ctrl.dead, [rabbit])
        st = w.history[-1]
        self.assertEqual((st["deaths_starvation"], st["deaths_rabbit"], st["count_rabbit"]),
                         (1, 1, 0))
        self.assertAlmostEqual(rabbit.reward_acc, config.REWARD_DEATH)

    def test_dehydration(self):
        w = empty_world()
        r, c = open_cell(w)
        fox = put(w, "fox", r, c, hydration=0.3)
        w.step(ctrls())
        self.assertFalse(fox.alive)
        self.assertEqual(fox.death_cause, "dehydration")
        self.assertEqual(w.history[-1]["deaths_dehydration"], 1)
        self.assertEqual(w.history[-1]["deaths_fox"], 1)

    def test_old_age(self):
        w = empty_world()
        r, c = open_cell(w)
        rabbit = put(w, "rabbit", r, c)
        rabbit.age = rabbit.lifespan - 1
        w.step(ctrls())
        self.assertEqual(rabbit.death_cause, "age")
        self.assertEqual(w.history[-1]["deaths_age"], 1)
        self.assertAlmostEqual(rabbit.reward_acc, config.REWARD_AGE_DEATH)   # 노화는 벌점 없음

    def test_lifespan_and_initial_age_are_spread(self):
        w = World(seed=9)
        for a in w.animals:
            m = w.params[a.species]["max_age"]
            self.assertTrue(0.8 * m - 1 <= a.lifespan <= 1.2 * m + 1)
            self.assertTrue(0 <= a.age <= 0.5 * m)
        self.assertGreater(len({a.age for a in w.animals}), 10)

    def test_predation_by_moving_onto_prey(self):
        old = FOX_RABBIT["pounce"]
        FOX_RABBIT["pounce"] = 1.0                                   # 덮치기가 반드시 성공하게
        self.addCleanup(FOX_RABBIT.__setitem__, "pounce", old)
        w = empty_world()
        r, c = open_cell(w)
        fox = put(w, "fox", r, c, energy=30.0, hydration=50.0)       # hunt_below(40) 미만 = 배고픔
        rabbit = put(w, "rabbit", r, c + 1)
        rctrl = Scripted()
        w.step(ctrls(rabbit=rctrl, fox=Scripted({fox.id: 4})))
        self.assertFalse(rabbit.alive)
        self.assertEqual(rabbit.death_cause, "predation")
        self.assertEqual(rctrl.dead, [rabbit])                       # died()는 정확히 1회
        self.assertEqual((fox.row, fox.col), (r, c + 1))
        self.assertIs(w.animal_at(r, c + 1), fox)
        self.assertIsNone(w.animal_at(r, c))
        cost = w.params["fox"]["move_cost"]
        self.assertAlmostEqual(fox.energy, 30.0 - cost + FOX_RABBIT["energy"])
        self.assertAlmostEqual(fox.hydration,
                               50.0 - w.params["fox"]["thirst"] + FOX_RABBIT["hydration"])
        st = w.history[-1]
        self.assertEqual((st["hunts"], st["deaths_predation"], st["deaths_rabbit"],
                          st["count_rabbit"], st["count_fox"]), (1, 1, 1, 0, 1))
        self.assertEqual((st["hunts_fox"], st["hunts_wolf"], st["deaths_predation_rabbit"],
                          st["deaths_predation_deer"]), (1, 0, 1, 0))
        # 욕구 비례 보상: 부족도 ≥ 0.5이면 base 전액
        self.assertAlmostEqual(fox.reward_acc, FOX_RABBIT["reward"] + config.REWARD_SURVIVE)
        w.step(ctrls())
        self.assertEqual(w.history[-1]["hunts_fox"], 0)              # hunts_*는 이번 step 값
        self.assertEqual(w.history[-1]["deaths_predation_rabbit"], 1)   # deaths_*는 누적

    def test_adjacent_catch_uses_catch_prob(self):
        for prob, expect_dead in ((1.0, True), (0.0, False)):
            with patched(FOX_RABBIT, catch=prob):
                w = empty_world()
                r, c = open_cell(w)
                fox = put(w, "fox", r, c, energy=30.0)
                rabbit = put(w, "rabbit", r, c + 1)
                w.step(ctrls())     # 둘 다 정지
                self.assertEqual(not rabbit.alive, expect_dead)
                self.assertEqual((fox.row, fox.col), (r, c))

    def test_fox_cannot_catch_deer(self):
        """여우는 사슴을 못 잡는다: 진입도 인접 포획도 안 되고, 사슴은 여우를 위험으로 보지 않는다."""
        with patched(FOX_RABBIT, catch=1.0):
            w = empty_world()
            r, c = open_cell(w)
            fox = put(w, "fox", r, c, energy=20.0)
            deer = put(w, "deer", r, c + 1)
            self.assertEqual(w.observe(fox)[2:4], (0, 0))            # 사슴은 먹이로 보이지 않는다
            self.assertEqual(w.observe(deer)[4:6], (0, 0))           # 여우는 사슴의 위험이 아니다
            self.assertFalse(w.can_move(fox, 4))
            for _ in range(5):
                w.step(ctrls(fox=Scripted({fox.id: 4})))
            self.assertTrue(deer.alive)
            self.assertEqual((fox.row, fox.col), (r, c))
            self.assertEqual(w.history[-1]["deaths_predation"], 0)

    def test_wolf_catches_both_deer_and_rabbit(self):
        for prey, hunt in (("deer", WOLF_DEER), ("rabbit", WOLF_RABBIT)):
            # 먹이 칸으로 진입(덮치기)
            with patched(hunt, pounce=1.0):
                w = empty_world()
                r, c = open_cell(w)
                wolf = put(w, "wolf", r, c, energy=40.0, hydration=50.0)
                victim = put(w, prey, r, c + 1)
                self.assertTrue(w.can_move(wolf, 4))
                w.step(ctrls(wolf=Scripted({wolf.id: 4})))
                self.assertFalse(victim.alive)
                self.assertEqual(victim.death_cause, "predation")
                self.assertEqual((wolf.row, wolf.col), (r, c + 1))
                self.assertAlmostEqual(
                    wolf.energy, min(100.0, 40.0 - w.params["wolf"]["move_cost"] + hunt["energy"]))
                st = w.history[-1]
                self.assertEqual((st["hunts_wolf"], st["hunts_fox"], st["hunts"]), (1, 0, 1))
                self.assertEqual(st[f"deaths_predation_{prey}"], 1)
                self.assertEqual(w.metrics[f"kills_wolf_{prey}"], 1)
                self.assertAlmostEqual(wolf.reward_acc, hunt["reward"] + config.REWARD_SURVIVE)
            # 인접 포획
            for prob, expect_dead in ((1.0, True), (0.0, False)):
                with patched(hunt, catch=prob):
                    w = empty_world()
                    r, c = open_cell(w)
                    put(w, "wolf", r, c, energy=30.0)
                    victim = put(w, prey, r, c + 1)
                    w.step(ctrls())
                    self.assertEqual(not victim.alive, expect_dead)

    def test_failed_pounce_keeps_predator_in_place(self):
        """덮치기 실패: 제자리에 머물고(막힌 이동 벌점 없음), 그 step에는 인접 포획도 없다."""
        with patched(WOLF_DEER, pounce=0.0, catch=1.0):
            w = empty_world()
            r, c = open_cell(w)
            wolf = put(w, "wolf", r, c, energy=40.0)
            deer = put(w, "deer", r, c + 1)
            w.step(ctrls(wolf=Scripted({wolf.id: 4})))
            self.assertTrue(deer.alive)
            self.assertEqual((wolf.row, wolf.col), (r, c))
            self.assertEqual(w.metrics["blocked_moves"], 0)
            self.assertAlmostEqual(wolf.energy, 40.0 - w.params["wolf"]["stay_cost"])

    def test_wolf_does_not_attack_fox(self):
        with patched(WOLF_DEER, catch=1.0), patched(WOLF_RABBIT, catch=1.0):
            w = empty_world()
            r, c = open_cell(w)
            wolf = put(w, "wolf", r, c, energy=20.0)
            fox = put(w, "fox", r, c + 1, energy=95.0)
            self.assertEqual(w.observe(wolf)[2:4], (0, 0))
            self.assertEqual(w.observe(wolf)[5], 0)
            w.step(ctrls(wolf=Scripted({wolf.id: 4})))
            self.assertTrue(fox.alive)
            self.assertEqual((wolf.row, wolf.col), (r, c))

    def test_large_prey_gives_more_energy_and_reward(self):
        self.assertGreater(WOLF_DEER["energy"], WOLF_RABBIT["energy"])
        self.assertGreater(WOLF_DEER["reward"], WOLF_RABBIT["reward"])
        self.assertGreater(FOX_RABBIT["pounce"], WOLF_RABBIT["pounce"])

    def test_full_fox_does_not_hunt(self):
        w = empty_world()
        r, c = open_cell(w)
        fox = put(w, "fox", r, c, energy=95.0)
        rabbit = put(w, "rabbit", r, c + 1)
        w.step(ctrls(fox=Scripted({fox.id: 4})))
        self.assertTrue(rabbit.alive)
        self.assertEqual((fox.row, fox.col), (r, c))                 # 한 칸에 한 마리

    def test_need_proportional_reward(self):
        # 배부른 토끼가 풀을 먹으면 보상이 작고, 배고픈 토끼는 base 전액을 받는다.
        got = {}
        for energy in (30.0, 85.0):
            w = empty_world()
            r, c = open_cell(w, lambda w, r, c: w.is_meadow(r, c))
            w.eaten[r * w.cols + c] = -10 * w.regrow                 # 다 자란 풀
            rabbit = put(w, "rabbit", r, c, energy=energy)
            w.step(ctrls())
            before = energy - w.params["rabbit"]["stay_cost"]
            self.assertAlmostEqual(rabbit.energy,
                                   min(100.0, before + w.params["rabbit"]["grass_energy"]))
            self.assertEqual(w.grass_level(r, c), 1 / w.regrow)      # 뜯긴 뒤 다시 자라는 중
            got[energy] = rabbit.reward_acc - config.REWARD_SURVIVE
        self.assertAlmostEqual(got[30.0], config.REWARD_GRASS)
        self.assertAlmostEqual(got[85.0], config.REWARD_GRASS * 2 * (100 - 84.5) / 100)
        self.assertLess(got[85.0], got[30.0])

    def test_flee_shaping(self):
        for action, expected in ((3, config.SHAPE_FLEE), (4, None)):
            w = empty_world()
            r, c = open_cell(w, lambda w, r, c: not w.is_meadow(r, c)
                             and not w.is_meadow(r, c - 1))
            rabbit = put(w, "rabbit", r, c)
            fox = put(w, "fox", r, c + 2, energy=95.0)               # 배불러서 사냥 안 함
            w.observe(rabbit)
            w._apply(rabbit, action, ctrls())
            if expected is not None:
                self.assertAlmostEqual(rabbit.reward_acc, expected + config.REWARD_SURVIVE)
            else:
                self.assertAlmostEqual(rabbit.reward_acc,
                                       config.SHAPE_APPROACH_DANGER + config.REWARD_SURVIVE)
            self.assertTrue(fox.alive)

    def test_observation(self):
        w = empty_world()
        r, c = open_cell(w)
        rabbit = put(w, "rabbit", r, c, energy=20.0, hydration=50.0)
        fox = put(w, "fox", r - 1, c, energy=80.0, hydration=10.0)
        e, h, grass, water, danger, dist = w.observe(rabbit)
        self.assertEqual((e, h, danger, dist), (0, 1, 1, 1))
        self.assertEqual(water, w.water_dir[r * w.cols + c])
        fe, fh, prey, pdist, fwater = w.observe(fox)
        self.assertEqual((fe, fh, prey, pdist), (2, 0, 2, 1))
        # 시야 밖이면 보이지 않는다
        w2 = empty_world()
        far = config.VISION["wolf"] + 2
        r, c = find_cell(w2, lambda r, c: 8 <= r < 40 and c + far < w2.cols
                         and all(not w2.is_water(r, c + i) for i in range(far)))
        rabbit = put(w2, "rabbit", r, c)
        fox = put(w2, "fox", r, c + config.VISION["rabbit"] + 1)
        self.assertEqual(w2.observe(rabbit)[4:], (0, 0))             # 토끼(4칸)는 못 봄
        self.assertEqual(w2.observe(fox)[2:4], (3, 2))               # 여우(5칸 이상)는 봄
        # 늑대(시야 8)는 여우(시야 6)가 못 보는 거리의 먹이도 본다
        w3 = empty_world()
        deer = put(w3, "deer", r, c)
        wolf = put(w3, "wolf", r, c + config.VISION["wolf"])
        obs = w3.observe(wolf)
        self.assertEqual((obs[2], obs[3], obs[5]), (3, 2, 2))        # 왼쪽 · 멂 · 대형
        self.assertEqual(w3.observe(deer)[4:6], (0, 0))              # 사슴(5칸)은 늑대를 못 봄
        w3b = empty_world()
        deer = put(w3b, "deer", r, c)
        put(w3b, "wolf", r, c + config.VISION["deer"])
        self.assertEqual(w3b.observe(deer)[4:6], (4, 2))             # 5칸: 보이고 '멂'
        rabbit = put(w3b, "rabbit", r, c + 1)
        self.assertEqual(w3b.observe(rabbit)[4:], (4, 2))            # 토끼(4칸)에게는 4칸 거리
        w4 = empty_world()
        rabbit = put(w4, "rabbit", r, c)
        fox = put(w4, "fox", r, c + config.VISION["wolf"])
        wolf = put(w4, "wolf", r, c + config.VISION["wolf"] + 1)
        self.assertEqual(w4.observe(fox)[2:4], (0, 0))               # 여우 시야 밖
        self.assertEqual(w4.observe(wolf)[5], 0)                     # 늑대 시야 밖
        self.assertTrue(rabbit.alive)

    def test_wolf_state_has_prey_size_and_prefers_large_at_same_distance(self):
        w = empty_world()
        r, c = find_cell(w, lambda r, c: 8 <= r < 40 and 8 <= c and c + 8 < w.cols
                         and all(not w.is_water(r + dr, c + dc)
                                 for dr in range(-3, 4) for dc in range(-3, 4)))
        wolf = put(w, "wolf", r, c, energy=50.0)
        self.assertEqual(len(w.observe(wolf)), 6)
        self.assertEqual(w.observe(wolf)[2:4] + w.observe(wolf)[5:], (0, 0, 0))   # 먹이 없음
        rabbit = put(w, "rabbit", r, c - 2)
        obs = w.observe(wolf)
        self.assertEqual((obs[2], obs[3], obs[5]), (3, 1, 1))        # 소형
        self.assertIs(wolf.target, rabbit)
        deer = put(w, "deer", r + 2, c)                              # 같은 거리(2칸)의 대형
        obs = w.observe(wolf)
        self.assertEqual((obs[2], obs[3], obs[5]), (2, 1, 2))        # 대형 우선
        self.assertIs(wolf.target, deer)
        near = put(w, "rabbit", r, c + 1)                            # 더 가까운 소형이 우선
        obs = w.observe(wolf)
        self.assertEqual((obs[2], obs[3], obs[5]), (4, 1, 1))
        self.assertIs(wolf.target, near)
        # 여우의 State에는 먹이 크기가 없다
        fox = put(w, "fox", r - 3, c)
        self.assertEqual(len(w.observe(fox)), 5)

    def test_herbivore_danger_is_nearest_of_all_its_predators(self):
        w = empty_world()
        r, c = find_cell(w, lambda r, c: 8 <= r < 40 and 8 <= c and c + 8 < w.cols
                         and all(not w.is_water(r + dr, c + dc)
                                 for dr in range(-4, 5) for dc in range(-4, 5)))
        rabbit = put(w, "rabbit", r, c)
        deer = put(w, "deer", r + 1, c)
        fox = put(w, "fox", r, c + 3, energy=95.0)
        self.assertEqual(w.observe(rabbit)[4:], (4, 2))              # 토끼: 여우(3칸, 오른쪽)
        self.assertEqual(w.observe(deer)[4:6], (0, 0))               # 사슴: 여우는 위험이 아님
        wolf = put(w, "wolf", r - 2, c, energy=95.0)
        self.assertEqual(w.observe(rabbit)[4:], (1, 1))              # 토끼: 더 가까운 늑대(2칸, 위)
        self.assertIs(rabbit.target, wolf)
        # 사슴: 늑대(3칸, 위). 늑대의 사냥 반경이 2칸이라 사슴에게는 3칸까지가 '가까움'이다.
        self.assertEqual(w.observe(deer)[4:6], (1, 1))
        self.assertEqual(config.NEAR_DIST["deer"], config.PARAMS["wolf"]["hunt_range"] + 1)
        self.assertTrue(fox.alive)

    def test_deer_grazes_two_cells(self):
        """사슴은 한 번에 자기 칸 + 인접 1칸의 풀을 뜯는다(토끼는 자기 칸만)."""
        for species, cells in (("deer", 2), ("rabbit", 1)):
            w = empty_world()
            r, c = open_cell(w)
            around = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            only_food_at(w, species, around)                         # 다 자란 먹이(토끼 풀 / 사슴 관목)
            self.assertEqual(w.params[species]["graze_extra"], cells - 1)
            a = put(w, species, r, c, energy=30.0)
            w.step(ctrls())
            p = w.params[species]
            self.assertAlmostEqual(a.energy, 30.0 - p["stay_cost"] + cells * p["grass_energy"])
            self.assertEqual(sum(1 for x in around if food_level(w, species, *x) < 0.5), cells)

    def test_no_predator_run_keeps_carnivores_absent(self):
        h, summary = run_population("rule", 3, 30, initial_counts=NO_PREDATORS)
        self.assertEqual((h[0]["count_fox"], h[0]["count_wolf"]), (0, 0))
        self.assertEqual((h[-1]["count_fox"], h[-1]["count_wolf"]), (0, 0))
        self.assertGreater(h[0]["count_rabbit"], 0)
        self.assertEqual(summary["deaths_predation"], 0)
        self.assertEqual(summary["coexisted"], 1)                    # 없는 종은 멸종으로 세지 않는다
        for s in config.SPECIES:
            for name in ("min", "max", "mean", "final", "extinct_step"):
                self.assertIn(f"{name}_{s}", summary)

    def test_train_mode_respawns_and_never_breeds(self):
        w = World(seed=21, mode="train", randomize=False)
        counts = {s: w.count(s) for s in config.SPECIES}
        ctrl = make_controllers("random", 21)
        deaths = 0
        for _ in range(150):
            w.step(ctrl)
            deaths += sum(w.deaths.values())
            self.assertEqual({s: w.count(s) for s in config.SPECIES}, counts)
            self.assertEqual(sum(w.births.values()), 0)
        self.assertGreater(deaths, 0)                                # 실제로 죽고 다시 채워짐

    def test_eval_mode_breeds_and_keeps_one_animal_per_cell(self):
        sim = Simulation(seed=8, policy_mode="rule")
        w = sim.world
        births = 0
        for _ in range(200):
            sim.step()
            births += w.history[-1]["births_rabbit"]
            cells = [(a.row, a.col) for a in w.animals]
            self.assertEqual(len(cells), len(set(cells)))
            for a in w.animals:
                self.assertTrue(a.alive)
                self.assertFalse(w.is_water(a.row, a.col))
                self.assertIs(w.animal_at(a.row, a.col), a)
                self.assertTrue(0 < a.energy <= 100 and 0 < a.hydration <= 100)
        self.assertGreater(births, 0)

    def test_breeding_costs_parent_energy(self):
        """에너지 예산 번식: 부모가 breed_cost를 내고, 새끼는 그 energy로 시작한다."""
        self.assertTrue(config.ENERGY_BUDGET_BREEDING)
        w = empty_world()
        r, c = open_cell(w, lambda w, r, c: not w.is_meadow(r, c))
        parent = put(w, "rabbit", r, c, energy=80.0, hydration=90.0)
        parent.cooldown = 0
        w.params["rabbit"]["breed_prob"] = 1.0
        w.step(ctrls())
        self.assertEqual(w.count("rabbit"), 2)
        child = [a for a in w.animals if a is not parent][0]
        cost = w.params["rabbit"]["breed_cost"]
        self.assertAlmostEqual(parent.energy, 80.0 - w.params["rabbit"]["stay_cost"] - cost)
        self.assertAlmostEqual(child.energy, cost)
        self.assertEqual(abs(child.row - r) + abs(child.col - c), 1)
        self.assertEqual(parent.cooldown, w.params["rabbit"]["breed_cooldown"])
        self.assertEqual(w.history[-1]["births_rabbit"], 1)


class RefugeTest(unittest.TestCase):
    """2단계: 굴(토끼) · 기근 대사 · 영역성 · 외부 유입. (사슴용 덤불은 없앴다.)"""

    def test_burrow_terrain(self):
        for seed in range(4):
            w = World(seed=seed)
            self.assertGreater(len(w.burrow_cells), 0)
            land = len(w.land_cells)
            self.assertAlmostEqual(len(w.burrow_cells) / land, config.BURROW_FRACTION, delta=0.005)
            for r in range(w.rows):
                for c in range(w.cols):
                    k = r * w.cols + c
                    if w.is_water(r, c):
                        self.assertFalse(w.is_burrow(r, c))        # 물 칸에서는 항상 False
                    if w.is_burrow(r, c):
                        self.assertFalse(w.is_meadow(r, c))        # 풀이 없다
                        self.assertFalse(w.shore[k])               # 물도 마실 수 없다
                        self.assertEqual(w.grass_level(r, c), 0.0)
            self.assertFalse(w.is_burrow(-1, 0))
            # 처음에 굴에 들어가 있는 것은 굴을 쓰는 종(토끼)뿐이다
            for a in w.animals:
                if w.is_burrow(a.row, a.col):
                    self.assertEqual(a.species, "rabbit")

    def test_same_seed_gives_same_refuges(self):
        a, b = World(seed=5), World(seed=5)
        self.assertEqual(a.burrow, b.burrow)
        self.assertNotEqual(a.burrow, World(seed=6).burrow)

    def test_rabbit_in_burrow_cannot_be_caught_or_seen(self):
        with patched(FOX_RABBIT, catch=1.0), patched(WOLF_RABBIT, catch=1.0, pounce=1.0):
            for predator in ("fox", "wolf"):
                w = empty_world(cover=True)
                r, c = burrow_cell(w)
                rabbit = put(w, "rabbit", r, c)
                hunter = put(w, predator, r, c - 1, energy=30.0)
                self.assertEqual(w.observe(hunter)[2:4], (0, 0))         # 보이지 않는다
                self.assertFalse(w.can_move(hunter, 4))                  # 들어갈 수 없다
                for action in (4, 0, 4, 0):                              # 덮치기 시도 · 인접 포획 시도
                    w.step(ctrls(**{predator: Scripted({hunter.id: action})}))
                self.assertTrue(rabbit.alive)
                self.assertEqual((hunter.row, hunter.col), (r, c - 1))
                self.assertEqual(w.history[-1]["deaths_predation"], 0)
                self.assertEqual(w.metrics["blocked_moves"], 2)
                # 굴 밖으로 나오면 다시 잡힌다
                w2 = empty_world(cover=True)
                rabbit = put(w2, "rabbit", r, c - 1)
                put(w2, predator, r, c - 2, energy=30.0)
                w2.step(ctrls())
                self.assertFalse(rabbit.alive)

    def test_only_burrow_users_enter_burrows(self):
        w = empty_world(cover=True)
        r, c = burrow_cell(w)
        for species in ("deer", "fox", "wolf"):
            a = put(w, species, r, c - 1)
            self.assertFalse(w.can_move(a, 4))
            w._apply(a, 4, ctrls())
            self.assertEqual((a.row, a.col), (r, c - 1))
            self.assertIsNone(w.animal_at(r, c))
            w._kill(a, "age", ctrls())
        rabbit = put(w, "rabbit", r, c - 1)
        self.assertTrue(w.can_move(rabbit, 4))
        w._apply(rabbit, 4, ctrls())
        self.assertEqual((rabbit.row, rabbit.col), (r, c))
        self.assertEqual(w.burrow_count(), 1)
        other = put(w, "rabbit", r, c - 1)                               # 한 칸에 한 마리
        self.assertFalse(w.can_move(other, 4))
        w._apply(other, 4, ctrls())
        self.assertEqual((other.row, other.col), (r, c - 1))
        # 굴 안에서는 먹지도 마시지도 못한다
        e, h = rabbit.energy, rabbit.hydration
        w._apply(rabbit, 0, ctrls())
        self.assertAlmostEqual(rabbit.energy, e - w.params["rabbit"]["stay_cost"])
        self.assertAlmostEqual(rabbit.hydration, h - w.params["rabbit"]["thirst"])

    def test_no_animal_ever_sits_in_a_forbidden_burrow(self):
        sim = Simulation(seed=4, policy_mode="random")
        w = sim.world
        seen = 0
        for _ in range(150):
            sim.step()
            for a in w.animals:
                if w.is_burrow(a.row, a.col):
                    self.assertEqual(a.species, "rabbit")
                    seen += 1
        self.assertGreater(seen, 0)

    def test_refuge_direction_replaces_grass_direction(self):
        """위협이 보이고 굶주리지 않았으면 '풀 방향' 자리에 굴 방향이 들어간다(상태 수는 그대로)."""
        self.assertEqual(n_states("rabbit"), 3375)
        w = empty_world(cover=True)
        r, c = burrow_cell(w)
        rabbit = put(w, "rabbit", r, c - 2, energy=95.0)
        obs = w.observe(rabbit)
        self.assertEqual(rabbit.refuge, -1)                              # 위협이 없으면 굴을 보지 않는다
        fox = put(w, "fox", r, c - 4, energy=95.0)                       # 왼쪽 2칸(가까움)
        obs = w.observe(rabbit)
        self.assertEqual(obs[4:], (3, 1))
        self.assertEqual(obs[2], 4)                                      # 굴은 오른쪽
        self.assertEqual((rabbit.refuge, rabbit.refuge_dist), (r * w.cols + c, 2))
        # 굴 쪽으로 가면 도주 shaping, 반대로 가면 벌점
        w._apply(rabbit, 4, ctrls())
        self.assertAlmostEqual(rabbit.reward_acc, config.SHAPE_FLEE + config.REWARD_SURVIVE)
        self.assertEqual(w.metrics["refuge_steps_rabbit"], 1)
        # rule 정책: 굴로 들어가고, 들어간 뒤에는 머문다
        rule = RulePolicy("rabbit", Random(0))
        self.assertEqual(rule.act(w, rabbit, w.observe(rabbit)), 4)
        w._apply(rabbit, 4, ctrls())
        self.assertTrue(w.is_burrow(rabbit.row, rabbit.col))
        obs = w.observe(rabbit)
        self.assertEqual(obs[2], 0)                                      # 이미 굴 안
        self.assertEqual(rule.act(w, rabbit, obs), 0)
        # 굶주렸고 위협이 멀면 풀 방향을 그대로 본다
        rabbit.energy = 20.0
        self.assertEqual(w.observe(rabbit)[5], 2)
        self.assertEqual(rabbit.refuge, -1)
        self.assertTrue(fox.alive)

    def test_thicket_is_gone(self):
        """덤불 지형과 관련 설정 · 속성이 없다. 사슴에게는 은신처가 없다."""
        for name in ("THICKET_USERS", "THICKET_FRACTION", "THICKET_CLUSTER", "THICKET_CATCH_MULT"):
            self.assertFalse(hasattr(config, name), name)
        w = World(seed=2)
        for name in ("is_thicket", "thicket", "thicket_cells", "cover"):
            self.assertFalse(hasattr(w, name), name)
        self.assertEqual(config.BURROW_USERS, ("rabbit",))
        deer = next(a for a in w.animals if a.species == "deer")
        self.assertFalse(deer.burrower)
        _, summary = run_population("rule", 3, 5)
        self.assertNotIn("thicket_cells", summary)
        self.assertNotIn("mean_in_refuge_deer", summary)
        self.assertIn("mean_in_refuge_rabbit", summary)

    def test_deer_grass_slot_is_always_grass_direction(self):
        """사슴 State의 '풀 방향' 자리는 위협이 보여도 먹이(관목) 방향이다(굴 방향이 들어가지 않는다)."""
        w = empty_world(cover=True)
        r, c = burrow_cell(w)                                            # 굴은 오른쪽(c)
        only_food_at(w, "deer", [(r, c - 3)])                            # 사슴의 먹이는 왼쪽 한 칸에만
        deer = put(w, "deer", r, c - 2, energy=95.0)
        self.assertEqual(w.observe(deer)[2], 3)
        put(w, "wolf", r - 2, c - 2, energy=95.0)                        # 위 2칸(가까움)
        obs = w.observe(deer)
        self.assertEqual(obs[4:6], (1, 1))
        self.assertEqual(obs[2], 3)                                      # 여전히 풀 방향(왼쪽)
        self.assertEqual(deer.refuge, -1)
        rabbit = put(w, "rabbit", r + 1, c - 2, energy=95.0)             # 토끼는 굴 방향을 본다
        w.observe(rabbit)
        self.assertGreaterEqual(rabbit.refuge, 0)

    def test_famine_metabolism(self):
        """기근 대사: energy가 FAMINE_BELOW 미만인 포식자는 비용이 famine_cost_mult배."""
        for species in config.CARNIVORES:
            mult = config.PARAMS[species]["famine_cost_mult"]
            self.assertLessEqual(mult, 1.0)
            w = empty_world()
            r, c = open_cell(w)
            a = put(w, species, r, c, energy=30.0)
            w.step(ctrls())
            self.assertAlmostEqual(a.energy, 30.0 - w.params[species]["stay_cost"] * mult)
            b = put(w, species, r, c + 1, energy=60.0)
            w.step(ctrls())
            self.assertAlmostEqual(b.energy, 60.0 - w.params[species]["stay_cost"])
        self.assertNotIn("famine_cost_mult", config.PARAMS["rabbit"])    # 초식에는 없다
        self.assertLess(config.FAMINE_BELOW, config.PARAMS["fox"]["breed_min_energy"])

    def test_territory_limit_suppresses_breeding(self):
        self.assertFalse(config.PARAMS["fox"].get("territory_limit"))    # 여우의 영역성 상한은 없앴다
        limit = config.PARAMS["wolf"]["territory_limit"]
        w = empty_world()
        r, c = open_cell(w)
        wolf = put(w, "wolf", r, c, energy=99.0, hydration=90.0)
        w.params["wolf"]["breed_prob"] = 1.0
        w.n_alive["wolf"] = limit                                        # 영역이 꽉 찼다
        for _ in range(20):
            wolf.cooldown = 0
            wolf.energy = 99.0
            w.step(ctrls())
        self.assertEqual(sum(1 for a in w.animals if a.species == "wolf"), 1)
        w.n_alive["wolf"] = 1
        w.params["wolf"]["territory_limit"] = 10 ** 9
        wolf.cooldown = 0
        wolf.energy = 99.0
        w.step(ctrls())
        self.assertEqual(sum(1 for a in w.animals if a.species == "wolf"), 2)

    def test_immigration_rescues_only_predators_that_started_present(self):
        old_enabled = config.IMMIGRATION_ENABLED
        old_rules = {s: dict(v) for s, v in config.IMMIGRATION.items()}
        try:
            for s in config.IMMIGRATION:
                config.IMMIGRATION[s]["prob"] = 1.0
            config.IMMIGRATION_ENABLED = True
            w = World(seed=3)
            w.clear_animals()
            w.step(ctrls())
            st = w.history[-1]
            self.assertEqual((st["count_fox"], st["count_wolf"]), (1, 1))
            self.assertEqual((st["immigrants_fox"], st["immigrants_wolf"]), (1, 1))
            self.assertEqual(st["count_rabbit"], 0)                      # 초식은 유입되지 않는다
            for a in w.animals:                                          # 가장자리로 들어온다
                self.assertTrue(a.row in (0, w.rows - 1) or a.col in (0, w.cols - 1))
            # 처음부터 없던 종(대조 실험)에는 유입이 없다
            h, summary = run_population("rule", 3, 20, initial_counts=NO_PREDATORS)
            self.assertEqual((h[-1]["count_fox"], h[-1]["count_wolf"]), (0, 0))
            self.assertEqual(summary["immigrants_wolf"], 0)
            # 끄면 유입이 없다
            config.IMMIGRATION_ENABLED = False
            w = World(seed=3)
            w.clear_animals()
            w.step(ctrls())
            self.assertEqual(w.history[-1]["count_wolf"], 0)
            # train 모드에는 적용하지 않는다
            config.IMMIGRATION_ENABLED = True
            w = World(seed=3, mode="train", randomize=False)
            n = w.count("wolf")
            w.step(ctrls())
            self.assertEqual((w.count("wolf"), w.metrics["immigrants_wolf"]), (n, 0))
        finally:
            config.IMMIGRATION_ENABLED = old_enabled
            for s, v in old_rules.items():
                config.IMMIGRATION[s].update(v)


class NicheTest(unittest.TestCase):
    """먹이 분리: 토끼는 풀(grazer), 사슴은 관목(browser)."""

    def test_browse_terrain(self):
        self.assertTrue(config.BROWSE_ENABLED)
        self.assertEqual(config.DIET, {"rabbit": "grass", "deer": "browse"})
        self.assertEqual(config.BROWSERS, ("deer",))
        w = World(seed=4)
        n = w.rows * w.cols
        self.assertEqual(len(w.browse_cells), sum(w.browse))
        self.assertGreaterEqual(len(w.browse_cells), 0.9 * config.BROWSE_COVERAGE * n)
        for r in range(w.rows):
            for c in range(w.cols):
                level = w.browse_level(r, c)
                self.assertTrue(0.0 <= level <= 1.0)
                if w.is_browse(r, c):                    # 풀밭 · 물 · 굴과 겹치지 않는다
                    self.assertFalse(w.is_meadow(r, c) or w.is_water(r, c) or w.is_burrow(r, c))
                else:
                    self.assertEqual(level, 0.0)
        self.assertFalse(w.is_browse(-1, 0))
        self.assertEqual(w.browse_level(-1, 0), 0.0)
        self.assertTrue(0.0 < w.stats()["browse_ratio"] <= 1.0)
        self.assertEqual(World(seed=4).browse, w.browse)                 # 같은 seed면 같은 배치
        for a in w.animals:                              # 처음에는 자기 먹이가 있는 칸에서 시작한다
            if a.species == "deer":
                self.assertTrue(w.is_browse(a.row, a.col))
                self.assertTrue(a.browser)
            elif a.species == "rabbit":
                self.assertTrue(w.is_meadow(a.row, a.col))
                self.assertFalse(a.browser)
            else:
                self.assertFalse(a.browser)
        self.assertGreater(config.BROWSE_REGROW_STEPS, config.GRASS_REGROW_STEPS)   # 관목이 더 느리다

    def test_each_herbivore_eats_only_its_own_food(self):
        """사슴은 풀을 먹지 않고, 토끼는 관목을 먹지 않는다."""
        for species, other in (("deer", "rabbit"), ("rabbit", "deer")):
            w = empty_world()
            r, c = open_cell(w)
            only_food_at(w, other, [(r, c)])             # 이 칸에는 '다른 종'의 먹이만 있다
            only_food_at(w, species, [])
            a = put(w, species, r, c, energy=30.0, hydration=50.0)
            w.step(ctrls())
            p = w.params[species]
            self.assertAlmostEqual(a.energy, 30.0 - p["stay_cost"])
            self.assertEqual(food_level(w, other, r, c), 1.0)            # 남의 먹이는 그대로
            self.assertEqual(w.observe(a)[2], 0)                         # 먹이 방향에도 안 잡힌다

    def test_deer_food_direction_points_to_browse(self):
        w = empty_world()
        r, c = wide_cell(w, 5)
        only_food_at(w, "deer", [(r, c + 3)])            # 관목은 오른쪽
        only_food_at(w, "rabbit", [(r, c - 2)])          # 풀은 왼쪽
        deer = put(w, "deer", r, c, energy=30.0)
        self.assertEqual(w.observe(deer)[2], 4)
        deer.alive = False
        w.occ[r * w.cols + c] = None
        rabbit = put(w, "rabbit", r, c, energy=30.0)
        self.assertEqual(w.observe(rabbit)[2], 3)
        w.browse_eaten[r * w.cols + c + 3] = w.turn      # 방금 뜯긴 관목은 먹이로 보이지 않는다
        deer2 = put(w, "deer", r + 1, c, energy=30.0)
        self.assertEqual(w.observe(deer2)[2], 0)

    def test_browse_regrows_slower_than_grass(self):
        w = empty_world()
        r, c = open_cell(w)
        only_food_at(w, "deer", [(r, c)])
        only_food_at(w, "rabbit", [(r, c + 1)])
        w.browse_eaten[r * w.cols + c] = w.turn
        w.eaten[r * w.cols + c + 1] = w.turn
        for _ in range(config.GRASS_REGROW_STEPS):
            w.step(ctrls())
        self.assertAlmostEqual(w.grass_level(r, c + 1), 1.0)
        self.assertAlmostEqual(w.browse_level(r, c),
                               config.GRASS_REGROW_STEPS / config.BROWSE_REGROW_STEPS)

    def test_browse_switch_off_restores_shared_grass(self):
        """BROWSE_ENABLED = False(BROWSERS가 빔)면 관목이 없고 사슴도 풀을 먹는다."""
        old = config.BROWSE_ENABLED, config.BROWSERS
        config.BROWSE_ENABLED, config.BROWSERS = False, ()
        try:
            w = World(seed=4)
            self.assertEqual(w.browse_cells, [])
            self.assertEqual(w.stats()["browse_ratio"], 0.0)
            w.clear_animals()
            r, c = open_cell(w)
            only_food_at(w, "deer", [(r, c)])
            self.assertTrue(w.is_meadow(r, c))
            deer = put(w, "deer", r, c, energy=30.0)
            self.assertFalse(deer.browser)
            w.step(ctrls())
            p = w.params["deer"]
            self.assertAlmostEqual(deer.energy, 30.0 - p["stay_cost"] + p["grass_energy"])
        finally:
            config.BROWSE_ENABLED, config.BROWSERS = old


class EnergyBudgetTest(unittest.TestCase):
    """에너지 예산 번식: 먹어서 모은 energy만큼만 새끼를 낸다(4종 공통)."""

    def test_config(self):
        self.assertTrue(config.ENERGY_BUDGET_BREEDING)
        for s in config.SPECIES:
            p = config.PARAMS[s]
            self.assertGreater(p["breed_cost"], 0.0)
            self.assertLess(p["breed_cost"], p["breed_min_energy"])

    def test_every_species_pays_breed_cost(self):
        for s in config.SPECIES:
            w = empty_world()
            w.meadow = [False] * (w.rows * w.cols)
            w.browse = [False] * (w.rows * w.cols)
            r, c = open_cell(w)
            parent = put(w, s, r, c, energy=99.0, hydration=90.0)
            parent.cooldown = 0
            w.params[s]["breed_prob"] = 1.0
            w.params[s]["territory_limit"] = None        # 영역성 확률 판정은 따로 테스트한다
            w.step(ctrls())
            p = w.params[s]
            self.assertEqual(w.count(s), 2, s)
            child = [a for a in w.animals if a is not parent][0]
            self.assertAlmostEqual(parent.energy, 99.0 - p["stay_cost"] - p["breed_cost"])
            self.assertAlmostEqual(child.energy, p.get("newborn_energy", p["breed_cost"]))
            self.assertEqual(child.cooldown, p.get("maturity", p["breed_cooldown"]))

    def test_no_breeding_below_min_energy(self):
        """energy가 breed_min_energy에 못 미치면(= 먹이가 부족하면) 번식하지 않는다."""
        for s in config.SPECIES:
            w = empty_world()
            w.meadow = [False] * (w.rows * w.cols)
            w.browse = [False] * (w.rows * w.cols)
            r, c = open_cell(w)
            parent = put(w, s, r, c, energy=w.params[s]["breed_min_energy"] - 1.0, hydration=90.0)
            parent.cooldown = 0
            w.params[s]["breed_prob"] = 1.0
            for _ in range(5):
                w.step(ctrls())
            self.assertEqual(w.count(s), 1, s)

    def test_second_litter_needs_refill(self):
        """한 번 번식하면 energy가 줄어, 다시 먹기 전에는 쿨다운이 끝나도 또 번식하지 못한다."""
        w = empty_world()
        r, c = open_cell(w)
        fox = put(w, "fox", r, c, energy=85.0, hydration=90.0)
        fox.cooldown = 0
        w.params["fox"]["breed_prob"] = 1.0
        w.step(ctrls())
        self.assertEqual(w.count("fox"), 2)
        for a in w.animals:
            a.cooldown = 0
            a.hydration = 90.0
        w.step(ctrls())
        self.assertEqual(w.count("fox"), 2)

    def test_newborn_energy_can_differ_from_cost(self):
        w = empty_world()
        r, c = open_cell(w)
        w.params["fox"]["newborn_energy"] = 25.0
        fox = put(w, "fox", r, c, energy=90.0, hydration=90.0)
        fox.cooldown = 0
        w.params["fox"]["breed_prob"] = 1.0
        w.step(ctrls())
        child = [a for a in w.animals if a is not fox][0]
        self.assertAlmostEqual(child.energy, 25.0)
        self.assertAlmostEqual(fox.energy, 90.0 - w.params["fox"]["stay_cost"]
                               - w.params["fox"]["breed_cost"])

    def test_switch_off_restores_halving(self):
        old = config.ENERGY_BUDGET_BREEDING
        config.ENERGY_BUDGET_BREEDING = False
        try:
            w = empty_world()
            r, c = open_cell(w, lambda w, r, c: not w.is_meadow(r, c))
            parent = put(w, "rabbit", r, c, energy=80.0, hydration=90.0)
            parent.cooldown = 0
            w.params["rabbit"]["breed_prob"] = 1.0
            w.step(ctrls())
            child = [a for a in w.animals if a is not parent][0]
            self.assertAlmostEqual(parent.energy, (80.0 - w.params["rabbit"]["stay_cost"]) / 2)
            self.assertAlmostEqual(child.energy, parent.energy)
        finally:
            config.ENERGY_BUDGET_BREEDING = old


class RangedHuntTest(unittest.TestCase):
    """늑대는 맨해튼 거리 2 이내의 먹이를 사냥 시도한다(여우는 인접만)."""

    def test_hunt_range_config(self):
        self.assertEqual(config.PARAMS["wolf"]["hunt_range"], 2)
        self.assertEqual(config.PARAMS["fox"].get("hunt_range", 1), 1)
        self.assertEqual(config.RANGED_HUNTERS, ("wolf",))
        for prey in config.PREY["wolf"]:
            hunt = config.HUNT[("wolf", prey)]
            self.assertIn("catch_far", hunt)
            self.assertLessEqual(hunt["catch_far"], hunt["catch"])      # 멀수록 쉽지는 않다
        w = World(seed=1)
        self.assertEqual(len(w._reach["fox"]), 4)
        self.assertEqual(len(w._reach["wolf"]), 12)                      # 인접 4 + 거리 2의 8칸
        self.assertTrue(all(abs(dr) + abs(dc) == dist <= 2 for dr, dc, dist in w._reach["wolf"]))

    def test_wolf_catches_prey_two_cells_away(self):
        for prey, hunt in (("deer", WOLF_DEER), ("rabbit", WOLF_RABBIT)):
            for offset in ((0, 2), (1, 1), (-2, 0), (-1, -1)):
                for prob, expect_dead in ((1.0, True), (0.0, False)):
                    with patched(hunt, catch_far=prob, catch=0.0):
                        w = empty_world()
                        r, c = wide_cell(w)
                        wolf = put(w, "wolf", r, c, energy=40.0, hydration=50.0)
                        victim = put(w, prey, r + offset[0], c + offset[1])
                        w.step(ctrls())                                  # 둘 다 정지
                        self.assertEqual(not victim.alive, expect_dead)
                        self.assertEqual((wolf.row, wolf.col), (r, c))   # 늑대는 제자리
                        st = w.history[-1]
                        self.assertEqual(st["far_hunts_wolf"], int(expect_dead))
                        self.assertEqual(w.metrics[f"far_kills_wolf_{prey}"], int(expect_dead))
                        self.assertEqual(w.metrics[f"kills_wolf_{prey}"], int(expect_dead))
                        if expect_dead:
                            self.assertEqual(victim.death_cause, "predation")
                            self.assertAlmostEqual(
                                wolf.energy, min(
                                    100.0, 40.0 - w.params["wolf"]["stay_cost"] + hunt["energy"]))
                            self.assertEqual(st["hunts_wolf"], 1)

    def test_adjacent_kill_is_not_counted_as_far(self):
        with patched(WOLF_DEER, catch=1.0, catch_far=0.0):
            w = empty_world()
            r, c = wide_cell(w)
            put(w, "wolf", r, c, energy=40.0)
            deer = put(w, "deer", r, c + 1)
            w.step(ctrls())
            self.assertFalse(deer.alive)
            self.assertEqual(w.metrics["far_hunts_wolf"], 0)
            self.assertEqual(w.metrics["kills_wolf_deer"], 1)

    def test_wolf_does_not_reach_three_cells_and_fox_only_adjacent(self):
        with patched(WOLF_DEER, catch=1.0, catch_far=1.0), patched(FOX_RABBIT, catch=1.0):
            w = empty_world()
            r, c = wide_cell(w)
            put(w, "wolf", r, c, energy=40.0)
            deer = put(w, "deer", r, c + 3)
            w.step(ctrls())
            self.assertTrue(deer.alive)
            w2 = empty_world()
            put(w2, "fox", r, c, energy=30.0)
            rabbit = put(w2, "rabbit", r, c + 2)
            diag = put(w2, "rabbit", r + 1, c + 1)
            for _ in range(3):
                w2.step(ctrls())
            self.assertTrue(rabbit.alive and diag.alive)                 # 여우는 2칸 거리를 못 잡는다
            self.assertNotIn("far_hunts_fox", w2.history[-1])

    def test_full_wolf_and_burrowed_rabbit_are_exempt_at_range(self):
        with patched(WOLF_DEER, catch_far=1.0), patched(WOLF_RABBIT, catch_far=1.0, catch=1.0):
            w = empty_world()
            r, c = wide_cell(w)
            put(w, "wolf", r, c, energy=95.0)                            # 배부르면 사냥하지 않는다
            deer = put(w, "deer", r, c + 2)
            w.step(ctrls())
            self.assertTrue(deer.alive)
            w2 = empty_world(cover=True)
            r, c = burrow_cell(w2)
            rabbit = put(w2, "rabbit", r, c)                             # 굴 안
            put(w2, "wolf", r, c - 2, energy=40.0)
            for _ in range(3):
                w2.step(ctrls())
            self.assertTrue(rabbit.alive)

    def test_no_ranged_kill_across_water_or_burrow(self):
        """거리 2 사냥은 사이 칸이 물 · 굴로 막혀 있으면 안 된다(대각선은 두 길 중 하나만 열려도 된다)."""
        with patched(WOLF_DEER, catch_far=1.0, catch=0.0):
            for kind in ("water", "burrow"):
                def block(w, r, c):
                    k = r * w.cols + c
                    if kind == "water":
                        w.water[k] = True
                    else:
                        w.burrow[k] = True
                # 직선 2칸: 사이 칸 하나가 막히면 못 잡는다
                w = empty_world()
                r, c = wide_cell(w)
                put(w, "wolf", r, c, energy=40.0)
                deer = put(w, "deer", r, c + 2)
                block(w, r, c + 1)
                self.assertFalse(w.open_path(r, c, 0, 2))
                for _ in range(3):
                    w.step(ctrls())
                self.assertTrue(deer.alive)
                self.assertEqual(w.metrics["far_hunts_wolf"], 0)
                # 대각선: 한쪽 길만 막히면 잡는다
                w = empty_world()
                put(w, "wolf", r, c, energy=40.0)
                deer = put(w, "deer", r + 1, c + 1)
                block(w, r, c + 1)
                self.assertTrue(w.open_path(r, c, 1, 1))
                w.step(ctrls())
                self.assertFalse(deer.alive)
                # 대각선: 두 길이 모두 막히면 못 잡는다
                w = empty_world()
                put(w, "wolf", r, c, energy=40.0)
                deer = put(w, "deer", r - 1, c - 1)
                block(w, r, c - 1)
                block(w, r - 1, c)
                self.assertFalse(w.open_path(r, c, -1, -1))
                for _ in range(3):
                    w.step(ctrls())
                self.assertTrue(deer.alive)
        w = empty_world()
        r, c = wide_cell(w)
        put(w, "rabbit", r, c + 1)                                       # 다른 동물은 길을 막지 않는다
        self.assertTrue(w.open_path(r, c, 0, 2))
        self.assertTrue(w.open_path(r, c, 0, 1))

    def test_wolf_picks_the_prey_it_is_most_likely_to_catch(self):
        """사정권에 먹이가 여럿이면 포획 확률이 가장 높은 것을 고른다."""
        with patched(WOLF_DEER, catch=0.0, catch_far=1.0), \
                patched(WOLF_RABBIT, catch=0.0, catch_far=0.0):
            w = empty_world()
            r, c = wide_cell(w)
            put(w, "wolf", r, c, energy=40.0)
            rabbits = [put(w, "rabbit", r, c + 1), put(w, "rabbit", r - 1, c),
                       put(w, "rabbit", r + 1, c + 1)]
            deer = put(w, "deer", r, c - 2)
            w.step(ctrls())
            self.assertFalse(deer.alive)
            self.assertTrue(all(x.alive for x in rabbits))


class SprintTest(unittest.TestCase):
    """사슴 질주: 같은 방향으로 2번 이상 연속 이동한 뒤 또 같은 방향이면 확률적으로 2칸."""

    def setUp(self):
        self.assertEqual(config.SPRINTERS, ("deer",))
        self.assertNotIn("sprint_prob", config.PARAMS["rabbit"])
        self.assertTrue(0.0 < config.PARAMS["deer"]["sprint_prob"] <= 1.0)
        self.assertGreaterEqual(config.PARAMS["deer"]["sprint_streak"], 2)

    def world(self, prob=1.0):
        w = empty_world()
        w.meadow = [False] * (w.rows * w.cols)           # 풀 · 관목을 없애 energy 계산을 단순하게
        w.browse = [False] * (w.rows * w.cols)
        w.params["deer"]["sprint_prob"] = prob
        return w

    def test_sprint_after_two_moves_in_the_same_direction(self):
        w = self.world()
        r, c = wide_cell(w, 6)
        deer = put(w, "deer", r, c - 5, energy=80.0)
        go = ctrls(deer=Scripted(default=4))
        w.step(go)
        self.assertEqual((deer.col, deer.last_dir, deer.streak), (c - 4, 4, 1))
        w.step(go)
        self.assertEqual((deer.col, deer.last_dir, deer.streak), (c - 3, 4, 2))
        self.assertEqual(w.metrics["sprints_deer"], 0)
        e = deer.energy
        w.step(go)                                       # 세 번째 같은 방향 → 2칸
        self.assertEqual((deer.row, deer.col, deer.streak), (r, c - 1, 3))
        self.assertIs(w.animal_at(r, c - 1), deer)
        self.assertIsNone(w.animal_at(r, c - 2))
        self.assertIsNone(w.animal_at(r, c - 3))
        p = w.params["deer"]
        self.assertAlmostEqual(deer.energy, e - p["move_cost"] - p["sprint_cost"])
        self.assertEqual(w.metrics["sprints_deer"], 1)
        self.assertEqual(w.history[-1]["sprints_deer"], 1)
        w.step(go)                                       # 계속 달리면 계속 2칸
        self.assertEqual(deer.col, c + 1)
        self.assertEqual(w.metrics["sprints_deer"], 2)
        self.assertEqual(w.metrics["sprints_threat_deer"], 0)

    def test_no_sprint_when_probability_is_zero_or_species_does_not_sprint(self):
        w = self.world(prob=0.0)
        r, c = wide_cell(w, 6)
        deer = put(w, "deer", r, c - 5)
        rabbit = put(w, "rabbit", r + 2, c - 5)
        go = ctrls(deer=Scripted(default=4), rabbit=Scripted(default=4))
        for _ in range(5):
            w.step(go)
        self.assertEqual(deer.col, c)                    # 한 번에 1칸씩
        self.assertEqual(rabbit.col, c)
        self.assertEqual(w.metrics["sprints_deer"], 0)
        self.assertEqual(w.metrics["sprints_rabbit"], 0)

    def test_turning_or_stopping_resets_the_streak(self):
        w = self.world()
        r, c = wide_cell(w, 6)
        deer = put(w, "deer", r, c)
        for action, where, streak in ((4, (r, c + 1), 1), (4, (r, c + 2), 2),
                                      (1, (r - 1, c + 2), 1),          # 방향을 바꾸면 1칸
                                      (1, (r - 2, c + 2), 2),
                                      (0, (r - 2, c + 2), 0),          # 멈추면 초기화
                                      (1, (r - 3, c + 2), 1), (1, (r - 4, c + 2), 2),
                                      (1, (r - 6, c + 2), 3)):         # 다시 2번 달린 뒤 질주
            w.step(ctrls(deer=Scripted(default=action)))
            self.assertEqual(((deer.row, deer.col), deer.streak), (where, streak))
        self.assertEqual(w.metrics["sprints_deer"], 1)

    def test_blocked_second_cell_moves_one_cell_only(self):
        for blocker in ("animal", "water", "burrow", "border"):
            w = self.world()
            r, c = wide_cell(w, 6)
            if blocker == "border":
                r, c = find_cell(w, lambda r, c: c == w.cols - 4 and all(
                    not w.is_water(r, c + i) for i in range(4))), None
                (r, c) = r
            deer = put(w, "deer", r, c)
            go = ctrls(deer=Scripted(default=4))
            w.step(go)
            w.step(go)
            self.assertEqual((deer.col, deer.streak), (c + 2, 2))
            k = r * w.cols + c + 4
            if blocker == "animal":
                put(w, "rabbit", r, c + 4)
            elif blocker == "water":
                w.water[k] = True
            elif blocker == "burrow":
                w.burrow[k] = True
            e = deer.energy
            w.step(go)
            self.assertEqual((deer.row, deer.col), (r, c + 3), blocker)   # 가능한 만큼(1칸)만
            self.assertAlmostEqual(deer.energy, e - w.params["deer"]["move_cost"])
            self.assertEqual(w.metrics["sprints_deer"], 0)
            self.assertEqual(deer.streak, 3)
            # 첫 칸부터 막히면 제자리이고 연속 횟수가 끊긴다
            if blocker in ("water", "border"):
                w.step(go)
                self.assertEqual((deer.row, deer.col), (r, c + 3))
                self.assertEqual((deer.streak, deer.last_dir), (0, 0))

    def test_sprint_under_threat_is_counted_and_rewarded_as_fleeing(self):
        w = self.world()
        r, c = wide_cell(w, 6)
        deer = put(w, "deer", r, c)
        deer.last_dir, deer.streak = 4, 2
        put(w, "wolf", r, c - 2, energy=95.0)            # 왼쪽 2칸, 배불러서 사냥 안 함
        w.observe(deer)
        w._apply(deer, 4, ctrls())
        self.assertEqual(deer.col, c + 2)
        self.assertEqual(w.metrics["sprints_threat_deer"], 1)
        self.assertEqual(w.metrics["flee_steps_deer"], 1)
        self.assertAlmostEqual(deer.reward_acc, config.SHAPE_FLEE + config.REWARD_SURVIVE)

    def test_deer_state_has_running_direction_only_under_threat(self):
        """사슴 State 끝 성분: 위협이 보이고 직전 step에 이동했으면 그 방향, 아니면 0."""
        w = self.world()
        r, c = wide_cell(w, 6)
        deer = put(w, "deer", r, c)
        go = ctrls(deer=Scripted({deer.id: 4}))
        w.step(go)
        self.assertEqual(deer.last_dir, 4)
        self.assertEqual(len(w.observe(deer)), 7)
        self.assertEqual(w.observe(deer)[4:], (0, 0, 0))                 # 위협이 없으면 0
        wolf = put(w, "wolf", r, c - 3, energy=95.0)                     # 왼쪽 4칸(이동 후 기준)
        self.assertEqual(w.observe(deer)[4:], (3, 2, 4))                 # 왼쪽 · 멂 · 오른쪽으로 달리는 중
        w.step(ctrls())                                                  # 멈추면 0
        self.assertEqual(w.observe(deer)[6], 0)
        self.assertTrue(wolf.alive)
        rabbit = put(w, "rabbit", r + 3, c)
        self.assertEqual(len(w.observe(rabbit)), 6)                      # 토끼 State는 그대로 6성분
        self.assertLess(ENCODERS["deer"](w.observe(deer)), n_states("deer"))

    def test_rule_deer_keeps_running_direction_when_fleeing(self):
        """규칙 기반 사슴: 거리를 똑같이 벌리는 방향이 여럿이면 달리던 방향을 유지한다."""
        w = self.world()
        r, c = wide_cell(w, 6)
        deer = put(w, "deer", r, c)
        put(w, "wolf", r - 2, c, energy=95.0)            # 위 2칸: 아래 · 왼쪽 · 오른쪽 모두 멀어진다
        for last in (2, 3, 4):
            deer.last_dir, deer.streak = last, 2
            for seed in range(10):
                rule = RulePolicy("deer", Random(seed))
                self.assertEqual(rule.act(w, deer, w.observe(deer)), last)
        deer.last_dir, deer.streak = 1, 2                # 달리던 방향이 늑대 쪽이면 유지하지 않는다
        picks = {RulePolicy("deer", Random(seed)).act(w, deer, w.observe(deer))
                 for seed in range(30)}
        self.assertEqual(picks, {2, 3, 4})
        # 규칙 사슴은 늑대가 3칸(사정권 + 1)에 오면 달아나기 시작한다
        w2 = self.world()
        deer = put(w2, "deer", r, c)
        put(w2, "wolf", r, c + 3, energy=95.0)
        self.assertEqual(w2.observe(deer)[4:6], (4, 1))
        self.assertIn(RulePolicy("deer", Random(0)).act(w2, deer, w2.observe(deer)), (1, 2, 3))

    def test_rule_deer_outruns_more_often_with_sprint(self):
        """질주가 있으면 규칙 사슴의 포식 사망이 줄어든다(고정 개체수 벤치마크)."""
        from food_chain.simulation import run_benchmark
        old = config.PARAMS["deer"]["sprint_prob"]
        try:
            out = {}
            for prob in (0.0, 1.0):
                config.PARAMS["deer"]["sprint_prob"] = prob
                out[prob] = sum(run_benchmark("rule", seed, 300)["deer_predation_per_1000"]
                                for seed in (1, 2, 3))
        finally:
            config.PARAMS["deer"]["sprint_prob"] = old
        self.assertLess(out[1.0], out[0.0])


class LearningTest(unittest.TestCase):
    def test_state_space_sizes(self):
        self.assertEqual(n_states("rabbit"), 3375)
        self.assertEqual(n_states("fox"), 675)
        seen = set()
        for i in range(n_states("rabbit")):
            obs = decode("rabbit", i)
            self.assertEqual(encode_rabbit(obs), i)
            seen.add(obs)
        self.assertEqual(len(seen), 3375)
        for i in range(n_states("fox")):
            self.assertEqual(encode_fox(decode("fox", i)), i)
        self.assertEqual(STATE_SHAPE["rabbit"], (3, 3, 5, 5, 5, 3))
        # 질주하는 초식(사슴)은 토끼 State 끝에 '달리는 방향'(5값)이 붙는다
        self.assertEqual(STATE_SHAPE["deer"], STATE_SHAPE["rabbit"] + (5,))
        self.assertEqual(STATE_SHAPE["fox"], (3, 3, 5, 3, 5))
        self.assertEqual(STATE_SHAPE["wolf"], (3, 3, 5, 3, 5, 3))        # 여우 State × 먹이 크기 3
        self.assertEqual(n_states("deer"), 16875)
        seen = set()
        for i in range(n_states("deer")):
            obs = decode("deer", i)
            self.assertEqual(ENCODERS["deer"](obs), i)
            seen.add(obs)
        self.assertEqual(len(seen), 16875)
        self.assertEqual(n_states("wolf"), 2025)
        seen = set()
        for i in range(n_states("wolf")):
            obs = decode("wolf", i)
            self.assertEqual(encode_wolf(obs), i)
            seen.add(obs)
        self.assertEqual(len(seen), 2025)
        self.assertIsNot(ENCODERS["deer"], ENCODERS["rabbit"])
        self.assertIs(ENCODERS["wolf"], encode_wolf)
        self.assertEqual(QTable("deer").algorithm, "sarsa")
        self.assertEqual(QTable("wolf").algorithm, "qlearning")

    def test_sarsa_update_rule(self):
        t = QTable("rabbit")
        self.assertEqual(t.algorithm, "sarsa")
        t.q[1][2] = 0.5
        t.q[7] = [1.0, 9.0, 0.0, 0.0, 0.0]
        t.update_sarsa(1, 2, r=0.3, s2=7, a2=0, alpha=0.1, gamma=0.95)
        # 최댓값(9.0)이 아니라 실제로 실행할 행동 a'=0의 값(1.0)을 쓴다
        self.assertAlmostEqual(t.q[1][2], 0.5 + 0.1 * (0.3 + 0.95 * 1.0 - 0.5))

    def test_qlearning_update_rule(self):
        t = QTable("fox")
        self.assertEqual(t.algorithm, "qlearning")
        t.q[1][2] = 0.5
        t.q[7] = [1.0, 9.0, 0.0, 0.0, 0.0]
        t.update_qlearning(1, 2, r=0.3, s2=7, alpha=0.1, gamma=0.95)
        self.assertAlmostEqual(t.q[1][2], 0.5 + 0.1 * (0.3 + 0.95 * 9.0 - 0.5))

    def test_terminal_update_has_no_bootstrap(self):
        t = QTable("rabbit")
        t.q[3][1] = 1.0
        t.update_terminal(3, 1, r=-5.0, alpha=0.1)
        self.assertAlmostEqual(t.q[3][1], 1.0 + 0.1 * (-5.0 - 1.0))

    def test_delayed_sarsa_uses_action_actually_taken(self):
        """다음 자기 차례에 고른 a'로 직전 전이를 갱신하고, 그 a'를 그대로 실행한다."""
        w = empty_world(mode="train")
        r, c = open_cell(w, lambda w, r, c: not w.is_meadow(r, c))
        rabbit = put(w, "rabbit", r, c)
        table = QTable("rabbit")
        learner = QPolicy(table, epsilon=1.0, learn=True, rng=Random(5), alpha=0.1, gamma=0.95)
        learner.sample_average = False                   # 갱신식 자체를 확인하기 위해 α 고정
        ctrl = ctrls(rabbit=learner)
        w.step(ctrl)
        self.assertEqual(table.entries(), 0)             # 첫 차례에는 갱신할 전이가 없다
        s0, a0, r0 = rabbit.prev_state, rabbit.prev_action, rabbit.reward_acc
        self.assertGreaterEqual(s0, 0)
        table.q[encode_rabbit(w.observe(rabbit))] = [0.1, 0.2, 0.3, 0.4, 0.5]
        s1 = encode_rabbit(w.observe(rabbit))
        w.step(ctrl)
        a1 = rabbit.prev_action                          # 실제로 실행한 a'
        expected = 0.1 * (r0 + 0.95 * [0.1, 0.2, 0.3, 0.4, 0.5][a1])
        if s0 == s1 and a0 == a1:                        # 같은 칸을 갱신한 경우
            expected = [0.1, 0.2, 0.3, 0.4, 0.5][a1] + 0.1 * (
                r0 + 0.95 * [0.1, 0.2, 0.3, 0.4, 0.5][a1] - [0.1, 0.2, 0.3, 0.4, 0.5][a1])
            self.assertAlmostEqual(table.q[s0][a0], expected)
        elif s0 != s1:
            self.assertAlmostEqual(table.q[s0][a0], expected)
        self.assertEqual(rabbit.prev_state, s1)

    def test_delayed_qlearning_uses_max(self):
        w = empty_world(mode="train")
        r, c = open_cell(w)
        fox = put(w, "fox", r, c, energy=95.0)
        table = QTable("fox")
        learner = QPolicy(table, epsilon=0.0, learn=True, rng=Random(5), alpha=0.1, gamma=0.95)
        learner.sample_average = False
        ctrl = ctrls(fox=Scripted({fox.id: 0}))
        w.step(ctrl)                                     # 정지 → 같은 상태 유지
        s = encode_fox(w.observe(fox))
        other = (s + 1) % n_states("fox")
        fox.prev_state, fox.prev_action, fox.reward_acc = other, 3, 0.25
        table.q[s] = [0.0, 0.0, 2.0, 0.0, 0.0]
        learner.act(w, fox, w.observe(fox))
        self.assertAlmostEqual(table.q[other][3], 0.1 * (0.25 + 0.95 * 2.0))
        self.assertEqual((fox.prev_state, fox.prev_action, fox.reward_acc), (s, 2, 0.0))

    def test_eaten_rabbit_gets_exactly_one_terminal_update(self):
        """다른 개체(여우) 차례에 잡아먹혀도 -5 terminal 갱신이 정확히 1회 들어간다."""
        calls = []

        class Spy(QTable):
            def update_terminal(self, s, a, r, alpha):
                calls.append((s, a, r))
                super().update_terminal(s, a, r, alpha)

        w = empty_world(mode="train")
        r, c = open_cell(w, lambda w, r, c: not w.is_meadow(r, c))
        rabbit = put(w, "rabbit", r, c)
        fox = put(w, "fox", r, c + 2, energy=30.0)
        table = Spy("rabbit")
        learner = QPolicy(table, epsilon=0.0, learn=True, rng=Random(1), alpha=0.1, gamma=0.95)
        learner.sample_average = False
        obs = w.observe(rabbit)
        learner.act(w, rabbit, obs)                      # 토끼 차례: (s, a) 기록
        w._apply(rabbit, 0, ctrls(rabbit=learner))
        s, a = rabbit.prev_state, rabbit.prev_action
        step_reward = rabbit.reward_acc
        # 여우 차례: 한 칸 다가온 뒤(인접 포획은 꺼 둠) 그다음 차례에 토끼 칸으로 진입
        ctrl = ctrls(rabbit=learner)
        with patched(FOX_RABBIT, catch=0.0, pounce=1.0):
            w.observe(fox)
            w._apply(fox, 3, ctrl)
            self.assertTrue(rabbit.alive)
            self.assertEqual(calls, [])
            w.observe(fox)
            w._apply(fox, 3, ctrl)
        self.assertFalse(rabbit.alive)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], (s, a))
        self.assertAlmostEqual(calls[0][2], step_reward + config.REWARD_DEATH)
        self.assertAlmostEqual(table.q[s][a], 0.1 * (step_reward + config.REWARD_DEATH))
        self.assertEqual(rabbit.prev_state, -1)
        learner.died(w, rabbit)                          # 다시 불려도 중복 갱신 없음
        self.assertEqual(len(calls), 1)
        self.assertEqual(w.count("rabbit"), 1)           # train 모드: 즉시 재생성

    def test_sample_average_learning_rate(self):
        """α = max(α, 1/n): 첫 방문은 표본 그대로, 이후 평균, 1/α번째부터는 고정 α."""
        table = QTable("fox")
        learner = QPolicy(table, learn=True, rng=Random(0), alpha=0.1, gamma=0.95)
        learner.sample_average = True
        self.assertEqual([round(learner._alpha(3, 1), 4) for _ in range(12)],
                         [1.0, 0.5, 0.3333, 0.25, 0.2, 0.1667, 0.1429, 0.125, 0.1111,
                          0.1, 0.1, 0.1])
        self.assertEqual(table.visits[3][1], 12)
        learner.sample_average = False
        self.assertEqual(learner._alpha(3, 2), 0.1)

    def test_eval_does_not_change_q(self):
        tables = {s: QTable(s) for s in config.SPECIES}
        rng = Random(0)
        for t in tables.values():
            for row in t.q:
                for i in range(5):
                    row[i] = rng.uniform(-1, 1)
        before = copy.deepcopy({s: t.q for s, t in tables.items()})
        sim = Simulation(seed=6, policy_mode="trained", tables=tables)
        for _ in range(150):
            sim.step()
        self.assertEqual(before, {s: t.q for s, t in tables.items()})
        self.assertGreater(sim.world.turn, 0)

    def test_epsilon_schedule_and_plan(self):
        steps = plan(100, "alternating")
        self.assertEqual(len(steps), 100)
        self.assertEqual(steps[0][0], {"rabbit", "deer"})    # ① 초식 2종 학습
        self.assertTrue(steps[0][2])                     #    육식은 규칙 기반
        self.assertAlmostEqual(steps[0][1], config.EPSILON_START)
        self.assertAlmostEqual(steps[39][1], config.EPSILON_END)
        self.assertAlmostEqual(steps[0][3], config.ALPHA)
        self.assertAlmostEqual(steps[39][3], config.ALPHA_END)
        self.assertEqual(steps[40][0], {"fox", "wolf"})      # ② 육식 2종 학습
        self.assertFalse(steps[40][2])
        self.assertEqual({frozenset(x[0]) for x in steps[80:]},
                         {frozenset({"rabbit", "deer"}), frozenset({"fox", "wolf"})})
        joint = plan(10, "joint")
        self.assertTrue(all(x[0] == set(config.SPECIES) for x in joint))
        self.assertAlmostEqual(joint[-1][1], config.EPSILON_END)

    def test_training_updates_all_four_tables_and_roundtrips_json(self):
        tables, rows, _ = train(episodes=5, steps=40, seed=1, save=False, out=None)
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(rows[0]), LOG_COLUMNS)
        self.assertEqual(set(tables), set(config.SPECIES))
        for s in config.SPECIES:
            self.assertGreater(tables[s].entries(), 0, s)
            self.assertEqual(len(tables[s].q), n_states(s))
        old = config.Q_TABLE_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                config.Q_TABLE_DIR = Path(d)
                for s, t in tables.items():
                    path = t.save()
                    self.assertEqual(path, Path(d) / f"{s}.json")    # data/q_tables/<종>.json
                    loaded = QTable.load(s)
                    self.assertEqual(loaded.algorithm, config.ALGORITHM[s])
                    for a, b in zip(loaded.q, t.q):
                        for x, y in zip(a, b):
                            self.assertAlmostEqual(x, y, places=5)
                self.assertEqual(sorted(p.name for p in Path(d).iterdir()),
                                 ["deer.json", "fox.json", "rabbit.json", "wolf.json"])
                sim = Simulation(seed=2, policy_mode="trained")      # 4개를 모두 읽어 실행
                for _ in range(20):
                    sim.step()
                # 상태 구성이 다른 표는 거부한다(여우 표를 늑대로 읽기)
                with self.assertRaises(ValueError):
                    QTable.load("wolf", Path(d) / "fox.json")
        finally:
            config.Q_TABLE_DIR = old

    def test_train_mode_uses_train_counts(self):
        """train 모드에서는 평가 때 수가 적은 육식을 더 많이 둔다."""
        w = World(seed=1, mode="train", randomize=False)
        for s in config.SPECIES:
            self.assertEqual(w.count(s), config.TRAIN_COUNT[s])
        self.assertGreater(config.TRAIN_COUNT["wolf"], config.PARAMS["wolf"]["initial_count"])
        self.assertGreater(config.TRAIN_COUNT["fox"], config.PARAMS["fox"]["initial_count"])

    def test_policies_return_valid_actions(self):
        w = World(seed=13)
        for mode in ("random", "rule"):
            ctrl = make_controllers(mode, 13)
            for a in w.animals:
                self.assertIn(ctrl[a.species].act(w, a, w.observe(a)), range(5))
        self.assertEqual(set(make_controllers("rule", 1)), set(config.SPECIES))
        self.assertIsInstance(make_controllers("random", 1)["wolf"], RandomPolicy)
        self.assertIsInstance(make_controllers("rule", 1)["wolf"], RulePolicy)


if __name__ == "__main__":
    unittest.main()
