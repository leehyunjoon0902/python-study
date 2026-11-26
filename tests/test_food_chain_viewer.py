"""뷰어(food_chain.viewer)·차트(food_chain.plots) 테스트.

코어(Simulation)가 없어도 돌도록, 인터페이스 계약 v2의 API를 그대로 흉내 낸
FakeSimulation 을 이 파일 안에 둔다(기본 4종 · 50×75 · 은신처 있음).

실행:
    uv run --no-sync python -m unittest tests.test_food_chain_viewer -v
    (산출물 위치를 지정하려면 환경변수 FOOD_CHAIN_TEST_OUT=<폴더>)
"""

from __future__ import annotations

import csv
import math
import os
import random
import tempfile
import time
import tkinter as tk
import unittest

SPECIES4 = ('rabbit', 'deer', 'fox', 'wolf')
SPECIES2 = ('rabbit', 'fox')
SPECIES = SPECIES4
PREY = {'fox': ('rabbit',), 'wolf': ('deer', 'rabbit')}
CAUSE_KEYS = ('deaths_starvation', 'deaths_dehydration', 'deaths_predation', 'deaths_age')
DEFAULT_COUNTS = {'rabbit': 150, 'deer': 80, 'fox': 40, 'wolf': 30}   # 합계 300


def stat_keys(species=SPECIES4):
    keys = ['turn'] + [f'{p}_{s}' for s in species for p in
                       ('count', 'avg_energy', 'avg_hydration', 'births', 'deaths')]
    keys += ['grass_ratio', 'browse_ratio', *CAUSE_KEYS, 'hunts']
    keys += [f'deaths_predation_{s}' for s in species if s not in PREY]
    keys += [f'hunts_{s}' for s in species if s in PREY]
    return keys


STAT_KEYS = stat_keys()


def _out_dir(name):
    base = os.environ.get('FOOD_CHAIN_TEST_OUT') or tempfile.mkdtemp(prefix='food_chain_viewer_')
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------- 가짜 시뮬레이션
class FakeAnimal:
    def __init__(self, id_, species, row, col, rng):
        self.id, self.species, self.row, self.col = id_, species, row, col
        self.energy = rng.uniform(40, 100)
        self.hydration = rng.uniform(40, 100)
        self.age = 0
        self.alive = True
        self.streak = 0      # 같은 방향 연속 이동 횟수(사슴 질주 조건)
        self.last_dir = 0


class FakeWorld:
    """계약서의 World API(rows, cols, turn, grass_level, is_meadow, is_water,
    animals, stats, history)만 구현한 무작위 세계. 종 목록은 species 로 정한다."""

    def __init__(self, rng, rows=50, cols=75, counts=None, species=SPECIES4):
        self.rng, self.rows, self.cols, self.turn = rng, rows, cols, 0
        self.species = tuple(species)
        counts = dict(DEFAULT_COUNTS if counts is None else counts)
        self.water = set()
        for _ in range(8):   # 물웅덩이 몇 개
            r0, c0 = rng.randrange(rows), rng.randrange(cols)
            for dr in range(-1, 2):
                for dc in range(-2, 2):
                    if 0 <= r0 + dr < rows and 0 <= c0 + dc < cols:
                        self.water.add((r0 + dr, c0 + dc))
        self.burrows = set()
        if hasattr(self, 'is_burrow'):
            for _ in range(25):
                r, c = rng.randrange(rows), rng.randrange(cols)
                if (r, c) not in self.water:
                    self.burrows.add((r, c))
        self.browse_cells = set()
        if hasattr(self, 'is_browse'):
            for _ in range(12):   # 4×4 관목 군집 몇 개(풀밭 · 물 · 굴과 겹치지 않음)
                r0, c0 = rng.randrange(rows - 3), rng.randrange(cols - 3)
                for dr in range(4):
                    for dc in range(4):
                        cell = (r0 + dr, c0 + dc)
                        if cell not in self.water and cell not in self.burrows:
                            self.browse_cells.add(cell)
        self.meadow = {(r, c) for r in range(rows) for c in range(cols)
                       if (r, c) not in self.water and (r, c) not in self.burrows
                       and (r, c) not in self.browse_cells and rng.random() < 0.7}
        self.immigrants = {s: 0 for s in self.species if s in PREY}
        self.sprints = 0
        self.far_hunts = 0
        self.grass = {cell: rng.random() for cell in self.meadow}
        self.browse = {cell: rng.random() for cell in self.browse_cells}
        self.next_id = 1
        self.animals = []
        for sp in self.species:
            for _ in range(counts.get(sp, 0)):
                self._spawn(sp)
        self.cum = dict.fromkeys(CAUSE_KEYS, 0)
        self.pred_cum = {s: 0 for s in self.species if s not in PREY}
        self._reset_step_counts()
        self.history = [self.stats()]

    def _reset_step_counts(self):
        self.births = dict.fromkeys(self.species, 0)
        self.deaths = dict.fromkeys(self.species, 0)
        self.hunts_by = {s: 0 for s in self.species if s in PREY}
        self.hunts = 0

    def _spawn(self, sp, row=None, col=None):
        while row is None or (row, col) in self.water:
            row, col = self.rng.randrange(self.rows), self.rng.randrange(self.cols)
        a = FakeAnimal(self.next_id, sp, row, col, self.rng)
        self.next_id += 1
        self.animals.append(a)
        return a

    def grass_level(self, r, c):
        return self.grass.get((r, c), 0.0)

    def is_meadow(self, r, c):
        return (r, c) in self.meadow

    def is_water(self, r, c):
        return (r, c) in self.water

    def _near_water(self, r, c):
        return any((r + dr, c + dc) in self.water for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    def stats(self):
        st = {'turn': self.turn}
        for sp in self.species:
            group = [a for a in self.animals if a.species == sp]
            st[f'count_{sp}'] = len(group)
            st[f'avg_energy_{sp}'] = sum(a.energy for a in group) / len(group) if group else 0.0
            st[f'avg_hydration_{sp}'] = sum(a.hydration for a in group) / len(group) if group else 0.0
            st[f'births_{sp}'] = self.births[sp]
            st[f'deaths_{sp}'] = self.deaths[sp]
        st['grass_ratio'] = sum(self.grass.values()) / max(1, len(self.meadow))
        st.update(self.cum)
        st['hunts'] = self.hunts
        for s, v in self.pred_cum.items():
            st[f'deaths_predation_{s}'] = v
        for s, v in self.hunts_by.items():
            st[f'hunts_{s}'] = v
        if hasattr(self, 'is_browse'):      # 현재 코어: 관목 비율 · 외부 유입 · 질주 · 2칸 사냥 누적
            st['browse_ratio'] = sum(self.browse.values()) / max(1, len(self.browse))
            for s, v in self.immigrants.items():
                st[f'immigrants_{s}'] = v
            if 'deer' in self.species:
                st['sprints_deer'] = self.sprints
            if 'wolf' in self.species:
                st['far_hunts_wolf'] = self.far_hunts
        return st

    def _kill(self, a, cause):
        a.alive = False
        self.cum[cause] += 1
        self.deaths[a.species] += 1
        if cause == 'deaths_predation':
            self.pred_cum[a.species] = self.pred_cum.get(a.species, 0) + 1

    def step(self):
        rng = self.rng
        self.turn += 1
        self._reset_step_counts()
        for cell in self.grass:
            self.grass[cell] = min(1.0, self.grass[cell] + 0.01)
        for cell in self.browse:
            self.browse[cell] = min(1.0, self.browse[cell] + 0.008)
        prey_at = {}
        for a in self.animals:
            if a.species not in PREY:
                prey_at.setdefault((a.row, a.col), []).append(a)
        for a in list(self.animals):
            if not a.alive:
                continue
            action = rng.randrange(5)     # 0 정지, 1 위, 2 아래, 3 왼쪽, 4 오른쪽
            dr, dc = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))[action]
            r, c = a.row + dr, a.col + dc
            if action and 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.water:
                a.row, a.col = r, c
                a.streak = a.streak + 1 if action == a.last_dir else 1
                a.last_dir = action
            else:
                a.streak, a.last_dir = 0, 0
            if a.species == 'deer':
                b = self.browse.get((a.row, a.col), 0.0)
                if b > 0.5:
                    self.browse[(a.row, a.col)] = 0.0
            a.age += 1
            a.energy -= 0.9 if a.species in PREY else 1.2
            a.hydration -= 1.5
            if self._near_water(a.row, a.col):
                a.hydration = 100.0
            if a.species not in PREY:
                g = self.grass.get((a.row, a.col), 0.0)
                if g > 0.2:
                    self.grass[(a.row, a.col)] = g - 0.2
                    a.energy = min(100.0, a.energy + 10)
                if a.energy > 70 and rng.random() < 0.03:
                    self._spawn(a.species, a.row, a.col)
                    self.births[a.species] += 1
            else:
                prey = [p for p in prey_at.get((a.row, a.col), [])
                        if p.alive and p.species in PREY[a.species]]
                if prey:
                    self._kill(prey[0], 'deaths_predation')
                    self.hunts += 1
                    self.hunts_by[a.species] += 1
                    a.energy = min(100.0, a.energy + 40)
                    if rng.random() < 0.2:
                        self._spawn(a.species, a.row, a.col)
                        self.births[a.species] += 1
            if a.energy <= 0:
                self._kill(a, 'deaths_starvation')
            elif a.hydration <= 0:
                self._kill(a, 'deaths_dehydration')
            elif a.age > 400:
                self._kill(a, 'deaths_age')
        self.animals = [a for a in self.animals if a.alive]
        if hasattr(self, 'is_browse'):      # 포식자가 적으면 가끔 외부에서 1마리 유입
            for s in self.immigrants:
                if sum(a.species == s for a in self.animals) < 3 and rng.random() < 0.05:
                    self._spawn(s)
                    self.immigrants[s] += 1
        self.history.append(self.stats())


class FakeBurrowWorld(FakeWorld):
    """2단계 코어처럼 은신처(is_burrow)가 있는 세계."""

    def is_burrow(self, r, c):
        return (r, c) in self.burrows


class FakeBrowseWorld(FakeBurrowWorld):
    """현재 코어처럼 굴(is_burrow) · 관목(is_browse, browse_level) · 외부 유입 · 질주 누적이 있는 세계."""

    def is_browse(self, r, c):
        return (r, c) in self.browse_cells

    def browse_level(self, r, c):
        return self.browse.get((r, c), 0.0)


class FakeSimulation:
    """계약서의 Simulation API(reset, step, policy_mode, world)를 흉내 낸다.

    기본 = 4종 · 50×75 · 굴 · 관목 · 외부 유입 있음. browse=False 면 굴만 있는 2단계 세계,
    species=SPECIES2, burrow=False 로 예전 2종 세계.
    """

    def __init__(self, seed=42, policy_mode='random', counts=None, species=SPECIES4,
                 rows=50, cols=75, burrow=True, browse=True):
        self.policy_mode = policy_mode
        self._kw = dict(rows=rows, cols=cols, counts=counts, species=species)
        if not burrow:
            self._cls = FakeWorld
        else:
            self._cls = FakeBrowseWorld if browse else FakeBurrowWorld
        self.reset(seed)

    def reset(self, seed=None):
        self.seed = seed if seed is not None else random.randrange(2 ** 32)
        self.world = self._cls(random.Random(self.seed), **self._kw)

    def step(self):
        self.world.step()


class _Evt:
    def __init__(self, x, y):
        self.x, self.y = x, y


def _make_root():
    try:
        root = tk.Tk()
    except tk.TclError as e:  # 디스플레이 없음
        raise unittest.SkipTest(f'Tk 사용 불가: {e}')
    if os.environ.get('FOOD_CHAIN_TEST_HIDE'):
        root.withdraw()
    return root


# ---------------------------------------------------------------- 뷰어 테스트
class ViewerTest(unittest.TestCase):
    def test_cell_size_fits_screen(self):
        from food_chain.viewer import PANEL_W, compute_cell_size
        cell = compute_cell_size(50, 75, 1920, 1080)
        self.assertEqual(cell, 16)   # 격자 1200×800
        for rows, cols, sw, sh in ((50, 75, 1920, 1080), (50, 75, 1366, 768),
                                   (40, 60, 1920, 1080), (40, 60, 1366, 768)):
            c = compute_cell_size(rows, cols, sw, sh)
            self.assertLessEqual(cols * c + PANEL_W + 60, sw)
            self.assertLessEqual(rows * c + 240, sh)

    def test_window_fits_1920x1080(self):
        """실제 창 요구 크기(제목줄·작업 표시줄 여유 포함)가 1920×1080 안에 들어가는지."""
        from food_chain import viewer
        sim = FakeSimulation(seed=4)
        root = _make_root()
        cell = viewer.compute_cell_size(sim.world.rows, sim.world.cols, 1920, 1080)
        v = viewer.Viewer(root, sim, cell=cell, species=SPECIES4)
        try:
            root.update()
            w, h = root.winfo_reqwidth(), root.winfo_reqheight()
            print(f'\n[창 크기] 50×75, cell {cell}px: {w}×{h} (+ 제목줄 ~31, 작업 표시줄 ~48)')
            self.assertLessEqual(w, 1920)
            self.assertLessEqual(h + 31 + 48, 1080)
            # 오른쪽 패널이 왼쪽(격자) 높이를 넘지 않음 = 정보 패널이 넘치지 않음
            right = root.grid_slaves(row=0, column=1)[0]
            left = root.grid_slaves(row=0, column=0)[0]
            self.assertLessEqual(right.winfo_reqheight(), left.winfo_reqheight() + 5)
        finally:
            v.destroy()

    def test_smoke_draw_click_reset(self):
        from food_chain import viewer
        sim = FakeSimulation(seed=1)
        root = _make_root()
        v = viewer.Viewer(root, sim, species=SPECIES4)
        try:
            root.update()
            self.assertTrue(v.has_burrow)
            self.assertEqual(v.herbivores, ('rabbit', 'deer'))
            self.assertEqual(v.carnivores, ('fox', 'wolf'))
            for _ in range(5):
                v.step_once()
                root.update()
            self.assertEqual(sim.world.turn, 5)
            # 지형은 이미지 1장, 동물은 살아 있는 수만큼만 있어야 함(쌓이지 않음)
            self.assertEqual(len(v.canvas.find_withtag('terrain')), 1)
            self.assertEqual(len(v.canvas.find_withtag('animal')), len(sim.world.animals))
            # 굴 칸은 굴 색으로 칠해짐
            br, bc = next(iter(sim.world.burrows))
            self.assertEqual(v._cell_keys[br * sim.world.cols + bc], 'U')
            # 종별 마릿수는 두 줄 글자(초식 합계 / 육식 합계), 종별 배고픔·목마름 막대는 없다
            counts = v.count_label.cget('text')
            st = sim.world.stats()
            herb = st['count_rabbit'] + st['count_deer']
            carn = st['count_fox'] + st['count_wolf']
            self.assertIn(f'초식 합계 {herb}마리', counts)
            self.assertIn(f'육식 합계 {carn}마리', counts)
            for sp, name in (('rabbit', '토끼'), ('deer', '사슴'), ('fox', '여우'), ('wolf', '늑대')):
                self.assertIn(f'{name} {st[f"count_{sp}"]}', counts)
            self.assertEqual(len(counts.splitlines()), 2)
            self.assertFalse(hasattr(v, 'bars'))
            self.assertFalse(hasattr(v, '_bar_items'))
            for name in ('HUNGER_COLOR', 'THIRST_COLOR', 'BAR_ROW_H'):
                self.assertFalse(hasattr(viewer, name), name)
            self.assertIn('사슴', v.deaths_label.cget('text'))
            # 굴 칸 클릭 → 칸 정보
            v._on_click(_Evt(bc * v.cell + 1, br * v.cell + 1))
            if v._selected_id is None:
                self.assertIn('은신처', v.sel_label.cget('text'))
            # 관목 칸: 자란 정도에 따라 관목 색 단계로 칠해지고(풀과 다른 키), 점 무늬가 겹쳐지며,
            # 클릭하면 설명이 나온다
            self.assertTrue(v.has_browse)
            br2, bc2 = next(iter(sim.world.browse_cells))
            key = v._cell_keys[br2 * sim.world.cols + bc2]
            self.assertIsInstance(key, int)
            self.assertGreaterEqual(key, viewer.BROWSE_KEY0)
            self.assertEqual(key - viewer.BROWSE_KEY0,
                             round(sim.world.browse_level(br2, bc2) * viewer.BROWSE_STEPS))
            self.assertNotIn(viewer._key_color(key), viewer.GRASS_PALETTE)   # 풀과 다른 색
            self.assertIsNotNone(v._pattern_img)
            v._selected_id = None
            v._click_cell = (br2, bc2)
            v.render()
            self.assertIn('관목(사슴 먹이)', v.sel_label.cget('text'))
            # 관목이 뜯기면 색 단계가 바뀐다
            sim.world.browse[(br2, bc2)] = 0.0
            v.render()
            self.assertEqual(v._cell_keys[br2 * sim.world.cols + bc2], viewer.BROWSE_KEY0)
            # 정보 패널: 풀 · 관목 비율, 굴 안 개체 수(뷰어가 직접 셈), 질주 · 2칸 사냥 누적
            n_b = sum(1 for a in sim.world.animals
                      if a.species == 'rabbit' and (a.row, a.col) in sim.world.burrows)
            text = v.env_label.cget('text')
            st = sim.world.stats()
            self.assertIn(f'풀 {st["grass_ratio"] * 100:.0f}%', text)
            self.assertIn(f'관목(사슴 먹이) {st["browse_ratio"] * 100:.0f}%', text)
            self.assertIn(f'굴 안 토끼 {n_b}마리', text)
            self.assertIn('질주 가능 사슴', text)
            self.assertIn('늑대 2칸 사냥 누적', text)
            sim.world.immigrants['wolf'] = 7
            v.render()
            self.assertIn('외부 유입 누적 여우 0 · 늑대 7', v.env_label.cget('text'))
            # 질주 조건(같은 방향 연속 이동 ≥ sprint_streak)을 채운 사슴에만 표식이 붙는다
            for a in sim.world.animals:
                a.streak, a.last_dir = 0, 0
            v.render()
            self.assertFalse(v.canvas.find_withtag('sprint'))
            deer = next(a for a in sim.world.animals if a.species == 'deer')
            deer.streak, deer.last_dir = viewer.SPRINT_STREAK['deer'], 4
            rabbit = next(a for a in sim.world.animals if a.species == 'rabbit')
            rabbit.streak, rabbit.last_dir = 9, 4          # 토끼는 질주하지 않는다
            v._selected_id = deer.id
            v.render()
            self.assertEqual(len(v.canvas.find_withtag('sprint')), 1)
            self.assertEqual(v.sprinting_ids, {deer.id})
            self.assertIn('질주 가능', v.sel_label.cget('text'))
            self.assertIn('질주 가능 사슴 1마리', v.env_label.cget('text'))
            # 늑대 선택: 시야(8칸)와 사냥 반경(2칸) 경계를 함께 그린다
            wolf = next(a for a in sim.world.animals if a.species == 'wolf')
            v._selected_id = wolf.id
            v.render()
            sel = v.sel_label.cget('text')
            self.assertIn(f'시야 반경 {viewer.VISION["wolf"]}칸', sel)
            self.assertIn('사냥 반경 2칸', sel)
            dashed = [i for i in v.canvas.find_withtag('vision')
                      if v.canvas.type(i) == 'line' and v.canvas.itemcget(i, 'dash')]
            self.assertTrue(dashed)
            # 덤불은 없어졌다: 덤불 관련 이름이 뷰어에 남아 있지 않다
            for name in ('THICKET_COLOR', 'THICKET_USERS', 'THICKET_LEAF', 'THICKET_CATCH_MULT'):
                self.assertFalse(hasattr(viewer, name), name)
            self.assertFalse(hasattr(v, 'has_thicket'))
            # 개체 클릭 → 선택 + 시야 표시
            a = sim.world.animals[0]
            v._on_click(_Evt(a.col * v.cell + 2, a.row * v.cell + 2))
            self.assertIsNotNone(v._selected_id)
            self.assertTrue(v.canvas.find_withtag('vision'))
            self.assertIn('energy', v.sel_label.cget('text'))
            # 재생/정지 토글
            v.toggle()
            self.assertTrue(v.playing)
            t_end = time.time() + 0.5
            while time.time() < t_end:
                # update() 대신 이벤트를 하나씩 처리(타이머가 계속 와도 시간 제한이 걸리게)
                root.tk.dooneevent(tk._tkinter.DONT_WAIT)
                time.sleep(0.001)
            v.toggle()
            self.assertFalse(v.playing)
            self.assertGreater(sim.world.turn, 5)
            # 새 배치
            v.reset()
            self.assertEqual(sim.world.turn, 0)
            self.assertIsNone(v._selected_id)
            root.update()
            out = os.environ.get('FOOD_CHAIN_TEST_OUT')
            if out:   # 눈으로 확인할 스크린샷(창 캡처는 테스트 밖 스크립트에서)
                v.step_once()
                root.update()
        finally:
            v.destroy()

    def test_two_species_config(self):
        """예전 2종(토끼·여우) · 40×60 · 은신처 없음 설정에서도 뷰어가 동작한다."""
        from food_chain import viewer
        sim = FakeSimulation(seed=5, species=SPECIES2, rows=40, cols=60, burrow=False,
                             counts={'rabbit': 100, 'fox': 50})
        root = _make_root()
        v = viewer.Viewer(root, sim, species=SPECIES2)
        try:
            for _ in range(3):
                v.step_once()
            root.update()
            self.assertFalse(v.has_burrow)
            text = v.count_label.cget('text')
            self.assertIn('초식 합계', text)
            self.assertIn('육식 합계', text)
            self.assertNotIn('사슴', text)
            # 합계는 있는 종만 더한다
            st = sim.world.stats()
            self.assertEqual(v._series.values[viewer.HERB_TOTAL][-1], st['count_rabbit'])
            self.assertEqual(v._series.values[viewer.CARN_TOTAL][-1], st['count_fox'])
            g = v.open_graph()
            g.mode.set('pairs')
            g.request(force=True)
            self.assertEqual(g.pairs, [('fox', 'rabbit')])
            self.assertNotIn('U', v._cell_keys)
            self.assertEqual(len(v.canvas.find_withtag('animal')), len(sim.world.animals))
        finally:
            v.destroy()

    def test_burrow_only_world(self):
        """관목·유입 키가 없는 2단계 코어(굴만)에서는 관목 표시 없이 동작한다."""
        from food_chain import viewer
        sim = FakeSimulation(seed=6, browse=False)
        root = _make_root()
        v = viewer.Viewer(root, sim, species=SPECIES4)
        try:
            v.step_once()
            root.update()
            self.assertTrue(v.has_burrow)
            self.assertFalse(v.has_browse)
            self.assertFalse(any(isinstance(k, int) and k >= viewer.BROWSE_KEY0
                                 for k in v._cell_keys))
            self.assertIsNone(v._pattern_img)
            self.assertNotIn('관목', v.env_label.cget('text'))
            self.assertIn('굴 안 토끼', v.env_label.cget('text'))
            self.assertNotIn('유입', v.env_label.cget('text'))
        finally:
            v.destroy()

    def test_vision_and_hunt_range_defaults(self):
        """시야 · 사냥 반경 · 질주 조건이 코어 설정(config)과 같고, 기본값도 현재 모델 값이다."""
        from food_chain import config, viewer
        self.assertEqual(viewer.DEFAULT_VISION, {'rabbit': 4, 'deer': 5, 'fox': 6, 'wolf': 8})
        for sp in config.SPECIES:
            self.assertEqual(viewer.VISION[sp], config.VISION[sp])
        for sp in config.CARNIVORES:
            self.assertEqual(viewer.HUNT_RANGE[sp], config.PARAMS[sp].get('hunt_range', 1))
        self.assertEqual(viewer.HUNT_RANGE, {'fox': 1, 'wolf': 2})
        self.assertEqual(viewer.SPRINT_STREAK.get('deer'),
                         config.PARAMS['deer'].get('sprint_streak'))
        self.assertNotIn('rabbit', viewer.SPRINT_STREAK)

    def test_all_dead_auto_stop(self):
        from food_chain import viewer
        sim = FakeSimulation(seed=2, counts={'rabbit': 3})
        root = _make_root()
        v = viewer.Viewer(root, sim, species=SPECIES4)
        try:
            v.toggle()
            sim.world.animals.clear()       # 전원 사망 상황
            v.step_once()
            self.assertFalse(v.playing)
            self.assertIn('모든 동물', v.message.cget('text'))
            self.assertIn('늑대 0 (멸종)', v.count_label.cget('text'))
        finally:
            v.destroy()

    def test_population_chart_and_graph_window(self):
        """패널 그래프 = 초식 합계(왼쪽 · 실선) vs 육식 합계(오른쪽 · 점선), 전체 기간 기본.
        그래프 창(G): 같은 그래프 / 쌍별 전환, 범위 전환, 닫았다 다시 열기, 새 배치 때 초기화."""
        from food_chain import config, viewer
        sim = FakeSimulation(seed=9)
        root = _make_root()
        v = viewer.Viewer(root, sim, species=SPECIES4)
        try:
            for _ in range(450):
                sim.step()
            v.render()
            root.update()
            # 합계 열은 config.HERBIVORES / CARNIVORES 의 합
            st = sim.world.stats()
            ser = v._series
            self.assertEqual(ser.n, len(sim.world.history))
            self.assertEqual(ser.values[viewer.HERB_TOTAL][-1],
                             sum(st[f'count_{s}'] for s in config.HERBIVORES))
            self.assertEqual(ser.values[viewer.CARN_TOTAL][-1],
                             sum(st[f'count_{s}'] for s in config.CARNIVORES))
            # 누적합 이동 평균 = 직접 계산한 평균
            vals = ser.values[viewer.HERB_TOTAL]
            self.assertAlmostEqual(ser.smooth(viewer.HERB_TOTAL, 200, 50),
                                   sum(vals[150:251]) / 101)
            # 패널 그래프: 기본 = 전체(0부터), 축 제목 · 범례 · 실선/점선 두 선
            self.assertEqual(v.chart_range.get(), 'all')
            texts = ' '.join(v.chart.itemcget(i, 'text') for i in v.chart.find_all()
                             if v.chart.type(i) == 'text')
            self.assertIn('초식 수(마리) · 왼쪽 눈금', texts)
            self.assertIn('육식 수(마리) · 오른쪽 눈금', texts)
            self.assertIn('초식 합계(토끼+사슴) · 실선', texts)
            self.assertIn('육식 합계(여우+늑대) · 점선', texts)
            self.assertIn('0', texts.split())          # 가로축이 0부터
            heavy = [i for i in v.chart.find_all() if v.chart.type(i) == 'line'
                     and float(v.chart.itemcget(i, 'width')) == 2.0]
            self.assertEqual(len(heavy), 2)            # 이동 평균 선 2개(종별 선 없음)
            self.assertEqual(sum(1 for i in heavy if v.chart.itemcget(i, 'dash')), 1)
            # 범위 전환: 최근 300 이면 가로축이 그만큼만
            v.chart_range.set('300')
            v._render_chart()
            a, b, t_min = viewer._range_window(ser, '300')
            self.assertEqual((b - a, t_min), (300, 300))
            v.chart_range.set('all')
            # 그래프 창: G 와 같은 동작, 다시 열면 같은 창
            g = v.open_graph()
            self.assertIs(v.open_graph(), g)
            root.update()
            self.assertEqual(g.mode.get(), 'total')
            self.assertEqual(len(g.summaries), 1)
            self.assertIn('초식', g.summaries[0])
            g.mode.set('pairs')
            g.request(force=True)
            self.assertEqual(g.pairs, [('fox', 'rabbit'), ('wolf', 'deer')])   # config.PREY 첫 먹이
            self.assertEqual(len(g.summaries), 2)
            self.assertIn('토끼', g.summaries[0])
            for rng, _ in viewer.GRAPH_RANGES:
                g.range.set(rng)
                g.request(force=True)
            # 재생 중에는 시간 간격마다만 다시 그린다
            g.range.set('all')
            g.request(force=True)
            v.playing = True
            n0 = g.draws
            for _ in range(5):
                sim.step()
                v.render()
            self.assertLessEqual(g.draws - n0, 1)
            v.playing = False
            v.step_once()
            self.assertGreater(g.draws, n0)
            # 닫아도 뷰어는 계속 동작하고 다시 열 수 있다
            g.close()
            self.assertIsNone(v.graph_win)
            v.step_once()
            g2 = v.open_graph()
            self.assertIsNot(g2, g)
            # 새 배치: 시계열도 처음부터
            v.reset()
            self.assertEqual(sim.world.turn, 0)
            self.assertEqual(v._series.n, 1)
            self.assertEqual(g2.summaries[0].count('표시 0~0 step'), 1)
        finally:
            v.destroy()

    def test_long_history_frame_time(self):
        """history 가 16,000 step 이어도 render() · 그래프 창 다시 그리기가 느려지지 않는다."""
        from food_chain import viewer
        sim = FakeSimulation(seed=10, counts={'rabbit': 40, 'deer': 20, 'fox': 6, 'wolf': 4})
        root = _make_root()
        v = viewer.Viewer(root, sim, cell=16, species=SPECIES4)
        try:
            g = v.open_graph()
            root.update()
            base = sim.world.history[-1]
            rng = random.Random(0)
            for t in range(1, 16001):    # 긴 history 를 직접 만든다(진동하는 개체수)
                row = dict(base)
                row['turn'] = t
                for sp, amp in (('rabbit', 100), ('deer', 40), ('fox', 15), ('wolf', 4)):
                    row[f'count_{sp}'] = int(amp * (2 + math.sin(t / 300)) + rng.randint(0, 3))
                sim.world.history.append(row)
            sim.world.turn = 16000
            t0 = time.perf_counter()
            v.render()                    # 처음 한 번은 16,000 행을 캐시에 읽는다
            first = (time.perf_counter() - t0) * 1000
            render_ms, graph_ms = [], []
            for _ in range(20):
                sim.world.history.append(dict(sim.world.history[-1], turn=sim.world.turn + 1))
                sim.world.turn += 1
                v.render()
                render_ms.append(v.last_render_ms)
                g.request(force=True)
                graph_ms.append(g.last_draw_ms)
            render_ms.sort()
            graph_ms.sort()
            print(f'\n[긴 history] 16,000 step: 첫 render {first:.0f} ms, 이후 render() 중앙값 '
                  f'{render_ms[10]:.1f} ms, 그래프 창 다시 그리기 중앙값 {graph_ms[10]:.1f} ms')
            self.assertLess(render_ms[10], 100)
            self.assertLess(graph_ms[10], 150)
        finally:
            v.destroy()

    def test_render_speed(self):
        """50×75, 동물 300마리 기준 프레임 시간(ms) 측정. 10 step/s = 100ms 이내."""
        from food_chain import viewer
        sim = FakeSimulation(seed=3)
        root = _make_root()
        v = viewer.Viewer(root, sim, cell=16, species=SPECIES4)
        try:
            root.update()
            render_ms, frame_ms = [], []
            for _ in range(60):
                t0 = time.perf_counter()
                sim.step()
                # 동물 수를 정확히 300마리로 맞춘다(번식·사망 보정)
                while len(sim.world.animals) < 300:
                    sim.world._spawn(random.choice(SPECIES4))
                del sim.world.animals[300:]
                v.render()
                render_ms.append(v.last_render_ms)
                root.update()          # 실제 화면 반영까지 포함
                frame_ms.append((time.perf_counter() - t0) * 1000)
            render_ms.sort()
            frame_ms.sort()
            med_r, med_f = render_ms[len(render_ms) // 2], frame_ms[len(frame_ms) // 2]
            print(f'\n[렌더링 속도] 50×75, 동물 {len(sim.world.animals)}마리: '
                  f'render() 중앙값 {med_r:.1f} ms (최대 {render_ms[-1]:.1f}), '
                  f'step+render+update 중앙값 {med_f:.1f} ms (최대 {frame_ms[-1]:.1f})')
            self.assertLess(med_f, 100, '초당 10 step 을 유지하기에 너무 느림')
        finally:
            v.destroy()


# ---------------------------------------------------------------- 차트 테스트
def write_fake_results(results_dir, seeds=(1, 2, 3), steps=250, episodes=200,
                       species=SPECIES4):
    """FakeSimulation 으로 eval_<policy>.csv 와 train_log.csv 를 만든다."""
    os.makedirs(results_dir, exist_ok=True)
    keys = stat_keys(species)
    for k, policy in enumerate(('random', 'rule', 'trained')):
        with open(os.path.join(results_dir, f'eval_{policy}.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=['seed'] + keys, extrasaction='ignore')
            wr.writeheader()
            for seed in seeds:
                counts = {'rabbit': 60 + 20 * k, 'deer': 40, 'fox': 12, 'wolf': 8}
                sim = FakeSimulation(seed=seed * 10 + k, policy_mode=policy, species=species,
                                     counts={s: counts[s] for s in species})
                for row in sim.world.history[:1]:
                    wr.writerow({'seed': seed, **row})
                for _ in range(steps):
                    sim.step()
                    wr.writerow({'seed': seed, **sim.world.history[-1]})
    rng = random.Random(0)
    with open(os.path.join(results_dir, 'train_log.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['episode', 'epsilon'] + [f'mean_reward_{s}' for s in species]
                    + [f'mean_lifespan_{s}' for s in species] + ['hunts']
                    + [f'q_entries_{s}' for s in species])
        for ep in range(1, episodes + 1):
            eps = max(0.05, 1.0 * 0.985 ** ep)
            rewards = [round(-30 - 8 * i + (40 + 10 * i) * (1 - eps) + rng.gauss(0, 6), 3)
                       for i, _ in enumerate(species)]
            lifes = [round(80 + 20 * i + 100 * (1 - eps), 1) for i, _ in enumerate(species)]
            wr.writerow([ep, round(eps, 4)] + rewards + lifes + [rng.randrange(20)]
                        + [ep * (10 - i) for i, _ in enumerate(species)])


class PlotsTest(unittest.TestCase):
    def setUp(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('matplotlib 없음 (uv run 으로 실행하세요)')

    def _check(self, results):
        from food_chain import plots
        saved = plots.make_all(results)
        names = sorted(os.path.basename(p) for p in saved)
        print('\n[차트] ' + ', '.join(names) + f'\n       → {os.path.join(results, "plots")}')
        expected = {f'{k}_{p}.png' for k in ('population', 'phase', 'needs', 'oscillation')
                    for p in ('random', 'rule', 'trained')} | {'compare_policies.png',
                                                               'training_curve.png'}
        self.assertEqual(set(names), expected)
        for p in saved:
            self.assertGreater(os.path.getsize(p), 10_000)

    def test_make_all(self):
        results = _out_dir('results')
        write_fake_results(results)
        self._check(results)

    def test_compare_with_immigrants(self):
        """eval_summary.csv 에 immigrants_<종> 열이 있으면 정책 비교 그림에 유입 패널이 붙는다."""
        from food_chain import plots
        results = _out_dir('results_immigrants')
        write_fake_results(results, seeds=(1, 2), steps=60, episodes=30)
        cols = ['policy', 'seed'] + [f'immigrants_{s}' for s in SPECIES4]
        with open(os.path.join(results, 'eval_summary.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(cols)
            for k, p in enumerate(('random', 'rule', 'trained')):
                for seed in (1, 2):
                    wr.writerow([p, seed, 0, 0, 10 - 3 * k + seed, 20 - 6 * k])
        rows = plots._read_csv(os.path.join(results, 'eval_summary.csv'))
        means = plots._immigrant_means(rows, ['random', 'rule', 'trained'])
        self.assertEqual(set(means), {'fox', 'wolf'})       # 전부 0인 초식은 뺌
        self.assertAlmostEqual(means['fox']['random'], 11.5)
        self.assertIsNone(plots._immigrant_means(rows and [{'policy': 'x', 'seed': 1}], ['x']))
        self._check(results)

    def test_lv_charts(self):
        """lv_<policy>.csv · lv_summary.csv 가 있으면 lv_ 차트가 생기고, noimm/ 평가는 plots/noimm/ 에 그린다."""
        import math
        from food_chain import plots
        results = _out_dir('results_lv')
        write_fake_results(results, seeds=(1, 2), steps=60, episodes=30)
        write_fake_results(os.path.join(results, 'noimm'), seeds=(1,), steps=40, episodes=1)
        os.remove(os.path.join(results, 'noimm', 'train_log.csv'))
        cols = ['seed', 'turn', 'count_rabbit', 'count_deer', 'count_fox', 'count_wolf',
                'deaths_rabbit', 'deaths_deer', 'kills_fox_rabbit', 'kills_wolf_deer',
                'kills_wolf_rabbit']
        steps, period, lag = 3000, 600, 150
        for policy in ('rule', 'trained'):
            with open(os.path.join(results, f'lv_{policy}.csv'), 'w', newline='',
                      encoding='utf-8') as f:
                wr = csv.writer(f)
                wr.writerow(cols)
                for seed in (1000, 1001):
                    for t in range(steps + 1):
                        ph = 2 * math.pi * t / period
                        ph_q = 2 * math.pi * (t - lag) / period
                        wr.writerow([seed, t, round(200 + 100 * math.sin(ph)),
                                     round(100 + 30 * math.sin(ph)), round(30 + 15 * math.sin(ph_q)),
                                     round(8 + 3 * math.sin(ph_q)), 1, 0, 1, 0, 0])
        summary_cols = ['policy', 'seed', 'predator', 'prey', 'steps', 'extinct_step',
                        'measurable', 'period', 'lag', 'lag_ratio', 'xcorr_max']
        with open(os.path.join(results, 'lv_summary.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(summary_cols)
            for policy in ('rule', 'trained'):
                for seed in (1000, 1001):
                    wr.writerow([policy, seed, 'fox', 'rabbit', steps, -1, 1, period, lag,
                                 lag / period, 0.9])
                    wr.writerow([policy, seed, 'wolf', 'deer', steps, -1, 0, '', 1000, '', 0.3])
        saved = plots.make_all(results)
        names = {os.path.relpath(p, os.path.join(results, 'plots')).replace(os.sep, '/')
                 for p in saved}
        lv = {f'lv_{k}_{p}.png' for k in ('timeseries', 'seeds', 'phase')
              for p in ('rule', 'trained')} | {'lv_summary.png'}
        self.assertTrue(lv <= names, lv - names)
        self.assertIn('noimm/population_rule.png', names)
        self.assertIn('noimm/compare_policies.png', names)
        for p in saved:
            self.assertGreater(os.path.getsize(p), 10_000)
        # 회전 방향: 포식자가 1/4 주기 늦는 궤적은 반시계(부호 면적 > 0)
        xs = [200 + 100 * math.sin(2 * math.pi * t / period) for t in range(period)]
        ys = [30 + 15 * math.sin(2 * math.pi * (t - lag) / period) for t in range(period)]
        self.assertGreater(plots._signed_area(xs, ys), 0)

    def test_make_all_two_species(self):
        results = _out_dir('results_2species')
        write_fake_results(results, seeds=(1, 2), steps=120, episodes=60, species=SPECIES2)
        self._check(results)

    def test_missing_files_are_skipped(self):
        from food_chain import plots
        empty = _out_dir('empty_results')
        self.assertEqual(plots.make_all(empty), [])
        only_train = _out_dir('only_train')
        write_fake_results(only_train, seeds=(1,), steps=5, episodes=30)
        for p in ('random', 'rule', 'trained'):
            os.remove(os.path.join(only_train, f'eval_{p}.csv'))
        saved = plots.make_all(only_train)
        self.assertEqual([os.path.basename(p) for p in saved], ['training_curve.png'])

    def test_missing_columns(self):
        """종별 열 일부가 없어도(예: 늑대 열 없음, 풀 비율 없음) 예외 없이 진행."""
        from food_chain import plots
        results = _out_dir('partial_cols')
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(results, 'eval_rule.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['seed', 'turn', 'count_rabbit', 'count_deer', 'count_fox'])
            for t in range(50):
                wr.writerow([1, t, 50 + t % 7, 30, 5 + t % 3])
        saved = plots.make_all(results)
        self.assertTrue(any('population_rule' in p for p in saved))


if __name__ == '__main__':
    unittest.main(verbosity=2)
