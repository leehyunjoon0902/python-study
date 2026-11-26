"""먹이사슬 생태계 tkinter 뷰어 (종 수에 무관: config.SPECIES 를 순회).

사용법:
    from food_chain.simulation import Simulation
    from food_chain import viewer
    viewer.run(Simulation(seed=1, policy_mode='trained'))

조작: Space = 재생/정지, → = 한 step, R = 새 배치, G = 개체수 그래프 창,
      격자 클릭 = 개체 선택(시야 · 사냥 반경 표시).

오른쪽 패널: 종별 마릿수(초식 합계 / 육식 합계 두 줄), 먹이층 · 굴 · 질주 · 사냥 · 유입 · 누적 사망 요약,
개체수 그래프 1장(초식 합계 = 왼쪽 눈금 · 실선, 육식 합계 = 오른쪽 눈금 · 점선, 기본 = step 0부터 전체).
그래프 창(G): 같은 그래프를 크게, 또는 포식자–먹이 쌍별(config.PREY)로. 둘 다 표준 tkinter 캔버스로 그린다.
'최대' 속도는 한 프레임에 시뮬레이션을 여러 step 돌리고 한 번 그린다.

지형: 풀밭(흙색 → 진초록) · 관목(사슴 전용 먹이, 회황색 → 올리브 + 점 무늬) · 굴(보라) · 물(파랑) · 맨땅(회갈색).
질주 조건(같은 방향 연속 이동 ≥ sprint_streak)을 채운 사슴에는 진행 방향으로 작은 노란 화살표를 붙인다.

성능 메모(직접 측정, Windows Tk 8.6):
- 캔버스 사각형을 칸마다 두고 색이 바뀐 칸만 itemconfig 해도, 동물이 화면 전체에
  흩어져 있으면 Tk가 캔버스 거의 전체를 다시 그려서 프레임당 ~200ms가 걸렸다.
- 그래서 지형은 '1칸 = 1픽셀'인 작은 PhotoImage에 색이 바뀐 칸만 put 하고,
  이를 cell 배로 확대 복사(copy -zoom)한 큰 이미지 1장으로 그린다.
- 동물은 미리 만들어 둔 픽셀 스프라이트(PhotoImage)를 create_image 로 찍는다
  (타원 여러 개보다 3배 이상 빠름). 동물·시야는 태그로 지우고 매 프레임 다시 그린다.
"""

from __future__ import annotations

import time
import tkinter as tk
import traceback

try:  # 코어가 아직 없을 때(테스트용 가짜 시뮬레이션)도 뷰어가 뜨도록 안전하게 가져온다.
    from food_chain import config as _config
except Exception:  # pragma: no cover - 코어 미완성 시
    _config = None

# ---------------------------------------------------------------- 종 정보
KNOWN_ORDER = ('rabbit', 'deer', 'fox', 'wolf')
DEFAULT_LABEL = {'rabbit': '토끼', 'deer': '사슴', 'fox': '여우', 'wolf': '늑대'}
DEFAULT_VISION = {'rabbit': 4, 'deer': 5, 'fox': 6, 'wolf': 8}
DEFAULT_HUNT_RANGE = {'fox': 1, 'wolf': 2}
DEFAULT_HERBIVORES = ('rabbit', 'deer')
DEFAULT_PREY = {'fox': ('rabbit',), 'wolf': ('deer', 'rabbit')}

SPECIES = tuple(getattr(_config, 'SPECIES', ('rabbit', 'fox')))
SPECIES_NAME = {**DEFAULT_LABEL, **dict(getattr(_config, 'SPECIES_LABEL', {}) or {})}
VISION = {**DEFAULT_VISION, **dict(getattr(_config, 'VISION', {}) or {})}
PREY = dict(getattr(_config, 'PREY', None) or DEFAULT_PREY)
BURROW_USERS = tuple(getattr(_config, 'BURROW_USERS', None) or ('rabbit',))
_PARAMS = dict(getattr(_config, 'PARAMS', None) or {})
# 사냥 반경(맨해튼): 이동 후 이 거리 안의 먹이를 잡으려 한다(여우 1 = 인접, 늑대 2)
HUNT_RANGE = {**DEFAULT_HUNT_RANGE,
              **{s: int(p.get('hunt_range', 1) or 1) for s, p in _PARAMS.items() if s in PREY}}
# 질주하는 종과 질주에 필요한 연속 이동 횟수(사슴: 같은 방향 2번 이상 연속 이동 후)
SPRINTERS = tuple(getattr(_config, 'SPRINTERS', None) or ('deer',))
SPRINT_STREAK = {s: int(_PARAMS.get(s, {}).get('sprint_streak', 2) or 2) for s in SPRINTERS}
_HERB = tuple(getattr(_config, 'HERBIVORES', ()) or ())
_CARN = tuple(getattr(_config, 'CARNIVORES', ()) or ())


def is_carnivore(sp: str) -> bool:
    """육식 여부: config.CARNIVORES/HERBIVORES → config.PREY → 기본 표 순으로 판단."""
    if sp in _CARN:
        return True
    if sp in _HERB:
        return False
    if sp in PREY:
        return True
    return sp not in DEFAULT_HERBIVORES and sp in ('fox', 'wolf')


# ---------------------------------------------------------------- 설정값
DANGER = 35            # energy 또는 hydration 이 이 값보다 낮으면 위험 표시
PANEL_W = 430          # 오른쪽 정보 패널 너비(px)
MAX_CELL = 20          # 셀 크기 상한(작은 격자가 지나치게 커지지 않게)
SPEEDS = (2, 5, 10, 20, 0)   # step/s, 0 = 최대
DEFAULT_SPEED = 10
MAX_SPEED_BUDGET_MS = 60     # '최대' 속도에서 한 프레임에 시뮬레이션만 돌리는 시간(ms)

FONT = 'Malgun Gothic'
POLICY_NAME = {'trained': '학습된 정책', 'random': '무작위 정책', 'rule': '규칙 기반 정책'}
CAUSES = (('deaths_starvation', '아사(굶주림)'), ('deaths_dehydration', '탈수(목마름)'),
          ('deaths_age', '노화'), ('deaths_predation', '포식'))

# 색상
SOIL_RGB = (224, 207, 163)      # 풀이 없는 초지(흙색)
GRASS_RGB = (34, 120, 52)       # 풀이 가득한 초지(진초록)
GRASS_STEPS = 16                # 풀 농도 색 단계 수(단계가 바뀔 때만 다시 칠함)
BARE_COLOR = '#9c9389'          # 풀이 못 자라는 맨땅
WATER_COLOR = '#2f74d0'
BURROW_COLOR = '#5e3c99'        # 은신처(굴) — 흙·풀·물·맨땅 어느 것과도 다른 보라
# 관목(사슴 전용 먹이층): 회황색(다 뜯김) → 올리브(다 자람). 풀(흙색 → 채도 높은 초록)과
# 색 계열이 다르고, 칸마다 어두운 점 무늬(줄기)를 덧그려 색을 못 구분해도 무늬로 알 수 있다.
BROWSE_LOW_RGB = (196, 190, 168)
BROWSE_RGB = (122, 112, 28)
BROWSE_STEPS = 16
BROWSE_DOT = '#3b3608'
BROWSE_DOTS = ((0.22, 0.25), (0.68, 0.2), (0.45, 0.62), (0.15, 0.8), (0.8, 0.72))   # 칸 안 점 위치(비율)
BROWSE_KEY0 = 100               # 칸 색 키: 풀 = 0..GRASS_STEPS, 관목 = BROWSE_KEY0 + 0..BROWSE_STEPS
# 굴 안 동물: 몸 색을 굴 쪽 색으로 바꿔 '숨어 있음'을 보여 준다(모양은 그대로).
SHELTER_FILL = {'burrow': '#c7b3ee'}
SHELTER_OUTLINE = {'burrow': '#2e1a5c'}
SPRINT_COLOR = '#ffe066'        # 질주 조건을 채운 사슴의 진행 방향 화살표
SPRINT_EDGE = '#3d2a00'
HUNT_RANGE_COLOR = '#c92a2a'    # 선택한 포식자의 사냥 반경 경계선
# 지도 위 모양 색: 초식 = 밝은 색(흰색·크림), 육식 = 진한 난색(주황·적갈색)
ANIMAL_STYLE = {
    'rabbit': {'fill': '#ffffff', 'outline': '#3d3d3d', 'part': '#3d3d3d', 'eye': '#222222'},
    'deer': {'fill': '#f3d9a4', 'outline': '#5a3410', 'part': '#7a4a1c', 'eye': '#222222'},
    'fox': {'fill': '#ff7a1a', 'outline': '#5c2500', 'part': '#5c2500', 'eye': '#222222'},
    'wolf': {'fill': '#93361f', 'outline': '#2b0d06', 'part': '#d8c3b0', 'eye': '#ffd43b'},
}
SPRITE_SCALE = {'rabbit': 0.74, 'deer': 1.0, 'fox': 0.8, 'wolf': 1.0}   # 칸 대비 크기
DANGER_OUTLINE = '#e3001b'
# 종별 선 색(plots.py 와 같은 값, 그래프 창의 종 쌍별 보기): 초식 = 파랑·보라, 육식 = 주황·적갈
LINE_COLOR = {'rabbit': '#2a78d6', 'deer': '#7048c8', 'fox': '#eb6834', 'wolf': '#a3242f'}
_FALLBACK_LINE = ('#1baf7a', '#d6336c', '#495057', '#e8a400')
SELECT_COLOR = '#ffd43b'
BG = '#f1f3f5'


def _rgb(rgb) -> str:
    return '#%02x%02x%02x' % tuple(rgb)


GRASS_PALETTE = [
    _rgb(round(a + (b - a) * i / GRASS_STEPS) for a, b in zip(SOIL_RGB, GRASS_RGB))
    for i in range(GRASS_STEPS + 1)
]
BROWSE_PALETTE = [
    _rgb(round(a + (b - a) * i / BROWSE_STEPS) for a, b in zip(BROWSE_LOW_RGB, BROWSE_RGB))
    for i in range(BROWSE_STEPS + 1)
]


def _key_color(k) -> str:
    """칸 색 키 → 색. 문자열 = 고정 지형, 정수 = 풀 단계, BROWSE_KEY0 이상 = 관목 단계."""
    if isinstance(k, str):
        return _KEY_COLOR[k]
    if k >= BROWSE_KEY0:
        return BROWSE_PALETTE[k - BROWSE_KEY0]
    return GRASS_PALETTE[k]


def compute_cell_size(rows: int, cols: int, screen_w: int, screen_h: int) -> int:
    """격자 캔버스가 화면 너비의 약 2/3 이하를 차지하도록 셀 크기를 정한다.

    1920×1080, 50×75 기준 16px(격자 1200×800, 창 약 1660×940).
    세로 여유 240px = 창 제목줄(~31) + 작업 표시줄(~40~48) + 범례 2줄·조작줄·메시지·여백.
    """
    by_frac = (screen_w * 0.64) / cols
    by_width = (screen_w - PANEL_W - 60) / cols
    by_height = (screen_h - 240) / rows
    return max(5, min(MAX_CELL, int(min(by_frac, by_width, by_height))))


_KEY_COLOR = {'W': WATER_COLOR, 'B': BARE_COLOR, 'U': BURROW_COLOR}
_SHELTER_KEY = {'U': 'burrow'}


# ---------------------------------------------------------------- 스프라이트
def _seg_dist(u, v, a, b):
    """점 (u, v) 에서 선분 a-b 까지 거리."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((u - ax) * dx + (v - ay) * dy) / (dx * dx + dy * dy or 1)))
    return ((u - ax - t * dx) ** 2 + (v - ay - t * dy) ** 2) ** 0.5


def _sprite_mask(species, size):
    """스프라이트 모양: {(x, y): 'body' | 'part'}. 'part' = 귀·뿔·주둥이 등 다른 색 부분."""
    px = {}
    scale = SPRITE_SCALE.get(species, 0.85)
    lw = 1.1 / (size * scale) * 2       # 약 1픽셀 굵기(정규화 좌표)
    for y in range(size):
        for x in range(size):
            u = ((x + 0.5) / size * 2 - 1) / scale      # -1 ~ 1 (스프라이트 안)
            v = ((y + 0.5) / size * 2 - 1) / scale
            if abs(u) > 1.05 or abs(v) > 1.05:
                continue
            if species == 'rabbit':
                # 둥근 몸 + 위로 긴 귀 두 개
                body = (u / 0.66) ** 2 + ((v - 0.3) / 0.6) ** 2 <= 1
                ear = any(((u - ex) / 0.2) ** 2 + ((v + 0.42) / 0.52) ** 2 <= 1
                          for ex in (-0.3, 0.3))
                if body or ear:
                    px[x, y] = 'body'
            elif species == 'deer':
                # 갸름한 얼굴 + 옆으로 난 귀 + 위로 갈라진 뿔
                face = (u / 0.46) ** 2 + ((v - 0.38) / 0.6) ** 2 <= 1
                ear = any(((u - ex) / 0.26) ** 2 + ((v + 0.02) / 0.14) ** 2 <= 1
                          for ex in (-0.56, 0.56))
                antler = False
                for s in (-1, 1):
                    segs = (((0.2 * s, -0.12), (0.34 * s, -1.0)),
                            ((0.27 * s, -0.55), (0.82 * s, -0.86)))
                    if any(_seg_dist(u, v, a, b) <= lw for a, b in segs):
                        antler = True
                if face or ear:
                    px[x, y] = 'body'
                elif antler:
                    px[x, y] = 'part'
            elif species == 'fox':
                # 아래로 뾰족한 역삼각형 얼굴 + 위쪽 뾰족한 귀 두 개
                face = -0.5 <= v <= 0.95 and abs(u) <= 0.95 * (0.95 - v) / 1.45
                ear = -0.98 <= v < -0.5 and abs(abs(u) - 0.6) <= 0.32 * (v + 0.98) / 0.48
                if face:
                    px[x, y] = 'body'
                elif ear:
                    px[x, y] = 'part'
            elif species == 'wolf':
                # 넓적한 머리 + 뭉툭한 주둥이(밝은 색) + 곧게 선 귀
                if -0.48 <= v <= 0.95:
                    half = 0.95 if v < 0.2 else 0.95 - (v - 0.2) * 0.85
                    if abs(u) <= half:
                        px[x, y] = 'part' if (v > 0.38 and abs(u) < 0.36) else 'body'
                if (x, y) not in px and -1.0 <= v < -0.48:
                    if abs(abs(u) - 0.58) <= 0.34 * (v + 1.0) / 0.52:
                        px[x, y] = 'body'
            elif u * u + v * v <= 0.85:
                px[x, y] = 'body'
    return px


_EYES = {   # (u, v) — 스프라이트 정규화 좌표
    'rabbit': ((-0.25, 0.22), (0.25, 0.22)),
    'deer': ((-0.2, 0.28), (0.2, 0.28)),
    'fox': ((-0.25, -0.2), (0.25, -0.2)),
    'wolf': ((-0.33, -0.05), (0.33, -0.05)),
}


def make_sprite(master, species, size, danger, shelter=None):
    """동물 픽셀 스프라이트(배경 투명). 위험하면 빨간 굵은 테두리.

    shelter = 'burrow' 이면 몸 색·테두리를 굴 색으로 바꾼다(위험 테두리는 우선).
    """
    st = dict(ANIMAL_STYLE.get(species, {'fill': '#be4bdb', 'outline': '#333333',
                                         'part': '#333333', 'eye': '#222222'}))
    if shelter in SHELTER_FILL:
        st['fill'] = SHELTER_FILL[shelter]
        st['outline'] = SHELTER_OUTLINE[shelter]
    img = tk.PhotoImage(master=master, width=size, height=size)
    px = _sprite_mask(species, size)
    # 작은 모양(토끼)은 2px 테두리면 몸이 거의 다 빨개져서 1px 로 둔다
    border = 2 if danger and size * SPRITE_SCALE.get(species, 0.85) >= 12.5 else 1
    edge_col = DANGER_OUTLINE if danger else st['outline']
    for (x, y), part in px.items():
        edge = any((x + dx, y + dy) not in px
                   for dx in range(-border, border + 1) for dy in range(-border, border + 1)
                   if abs(dx) + abs(dy) <= border)
        if part == 'part' and species == 'deer':   # 가는 뿔은 전부 테두리라 뿔 색 유지
            col = st['part']
        elif edge:
            col = edge_col
        elif part == 'part':
            col = st['part']
        else:
            col = st['fill']
        img.put(col, to=(x, y))
    if size >= 12:   # 눈
        scale = SPRITE_SCALE.get(species, 0.85)
        for eu, ev in _EYES.get(species, ((-0.25, -0.1), (0.25, -0.1))):
            ex = int(((eu * scale) + 1) / 2 * size)
            ey = int(((ev * scale) + 1) / 2 * size)
            if (ex, ey) in px:
                img.put(st['eye'], to=(ex, ey))
    if shelter == 'burrow':
        # 굴 구멍: 몸 뒤(빈 픽셀)에 어두운 타원 → '구멍 속에 숨은 토끼'
        cx, cy, rx, ry = size / 2, size * 0.7, size * 0.5, size * 0.3
        for y in range(size):
            for x in range(size):
                if (x, y) not in px and ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1:
                    img.put(SHELTER_OUTLINE['burrow'], to=(x, y))
    return img


# ---------------------------------------------------------------- 뷰어
class Viewer:
    """mainloop 없이도 만들고 그리고 파괴할 수 있는 뷰어 본체(테스트용으로 분리).

    species: 표시할 종 순서. 생략하면 config.SPECIES. 그 밖의 종이 world 에 나타나면
    뒤에 덧붙인다(예외 없이 동작).
    """

    def __init__(self, root: tk.Misc, sim, cell: int | None = None, species=None):
        self.root = root
        self.sim = sim
        w = sim.world
        sp = list(species or SPECIES)
        for a in w.animals:
            if a.species not in sp:
                sp.append(a.species)
        self.species = tuple(sp)
        self.herbivores = tuple(s for s in self.species if not is_carnivore(s))
        self.carnivores = tuple(s for s in self.species if is_carnivore(s))
        self.line_color = dict(LINE_COLOR)
        for i, s in enumerate(s for s in self.species if s not in LINE_COLOR):
            self.line_color[s] = _FALLBACK_LINE[i % len(_FALLBACK_LINE)]
        if cell is None:
            cell = compute_cell_size(w.rows, w.cols,
                                     root.winfo_screenwidth(), root.winfo_screenheight())
        self.cell = cell
        self.playing = False
        self.speed = tk.IntVar(master=root, value=DEFAULT_SPEED)
        self._timer = None
        self._selected_id = None
        self._world_ref = None       # 지형 캐시가 어떤 World 기준인지
        self._cell_keys: list = []           # 칸별 현재 색 키(바뀐 칸만 다시 칠함)
        self._sprites: dict = {}              # (종, 위험, 크기) → PhotoImage
        self._step_times: list[float] = []   # 실제 진행 속도 측정용
        self.last_render_ms = 0.0
        self.has_burrow = callable(getattr(w, 'is_burrow', None))
        self.has_browse = (callable(getattr(w, 'is_browse', None))
                           and callable(getattr(w, 'browse_level', None)))
        self.shelter_counts: dict = {}       # (은신처, 종) → 마리 수(매 프레임 다시 셈)
        self.sprinting_ids: set = set()      # 질주 조건을 채운 개체 id(매 프레임 다시 셈)
        # 개체수 시계열 캐시(history 를 새 step 만큼만 읽는다) — 패널 그래프와 그래프 창이 함께 쓴다.
        # 초식 / 육식 합계는 config.HERBIVORES / CARNIVORES(is_carnivore)로 나눈 종들의 합이다.
        self._series = SeriesCache(
            [f'count_{s}' for s in self.species],
            derived={HERB_TOTAL: [f'count_{s}' for s in self.herbivores],
                     CARN_TOTAL: [f'count_{s}' for s in self.carnivores]})
        self.chart_range = tk.StringVar(master=root, value='all')   # 패널 그래프 범위: all | 2000 | 300
        self.graph_win = None                # 개체수 그래프 창(GraphWindow), 닫혀 있으면 None

        self._build_ui()
        self._bind_keys()
        self.render()

    # ------------------------------------------------------------ UI 구성
    def _build_ui(self):
        root = self.root
        w = self.sim.world
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title(f'먹이사슬 생태계 · {self._policy_label()}')
            root.configure(bg=BG)

        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky='n', padx=(10, 6), pady=10)
        right = tk.Frame(root, bg=BG, width=PANEL_W)
        right.grid(row=0, column=1, sticky='n', padx=(6, 10), pady=10)

        # 격자 캔버스
        cw, ch = w.cols * self.cell, w.rows * self.cell
        self.canvas = tk.Canvas(left, width=cw, height=ch, highlightthickness=1,
                                highlightbackground='#868e96', bg=BARE_COLOR)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self._on_click)
        self._create_cells()

        # 지도 범례
        self._build_map_legend(left)

        # 조작 버튼
        ctrl = tk.Frame(left, bg=BG)
        ctrl.pack(fill='x', pady=(6, 0))
        bopt = dict(font=(FONT, 10), takefocus=0, padx=8)
        self.play_btn = tk.Button(ctrl, text='▶ 재생 (Space)', width=14,
                                  command=self.toggle, **bopt)
        self.play_btn.pack(side='left')
        tk.Button(ctrl, text='한 step (→)', command=self.step_once, **bopt).pack(side='left', padx=4)
        tk.Button(ctrl, text='새 배치 (R)', command=self.reset, **bopt).pack(side='left')
        tk.Button(ctrl, text='개체수 그래프 창 (G)', command=self.open_graph,
                  **bopt).pack(side='left', padx=(4, 0))
        tk.Label(ctrl, text='   속도:', font=(FONT, 10), bg=BG).pack(side='left')
        for sp in SPEEDS:
            tk.Radiobutton(ctrl, text='최대' if sp == 0 else f'{sp}', value=sp,
                           variable=self.speed, indicatoron=0, takefocus=0,
                           font=(FONT, 9), padx=6, command=self._on_speed).pack(side='left')
        tk.Label(ctrl, text='step/s', font=(FONT, 10), bg=BG).pack(side='left', padx=(3, 0))

        self.message = tk.Label(left, text='', font=(FONT, 11, 'bold'), fg='#c92a2a',
                                bg=BG, anchor='w')
        self.message.pack(fill='x', pady=(4, 0))

        # ---------- 오른쪽 정보 패널
        self.header = tk.Label(right, text='', font=(FONT, 15, 'bold'), bg=BG,
                               anchor='w', justify='left')
        self.header.pack(fill='x')
        self.subheader = tk.Label(right, text='', font=(FONT, 9), fg='#495057',
                                  bg=BG, anchor='w', justify='left', wraplength=PANEL_W)
        self.subheader.pack(fill='x', pady=(0, 6))

        # 종별 마릿수(초식 합계 / 육식 합계) — 두 줄 글자
        self.count_label = tk.Label(right, text='', font=(FONT, 11), bg=BG, anchor='w',
                                    justify='left', wraplength=PANEL_W)
        self.count_label.pack(fill='x')

        # 먹이층(풀 · 관목 비율) · 굴 · 질주 · 2칸 사냥 · 유입 · 누적 사망(작은 글자)
        self.env_label = tk.Label(right, text='', font=(FONT, 9), bg=BG, fg='#495057',
                                  anchor='w', justify='left', wraplength=PANEL_W)
        self.env_label.pack(fill='x', pady=(4, 0))
        self.deaths_label = tk.Label(right, text='', font=(FONT, 9), bg=BG, fg='#495057',
                                     anchor='w', justify='left', wraplength=PANEL_W)
        self.deaths_label.pack(fill='x', pady=(2, 0))

        # 개체수 그래프: 초식 합계(왼쪽 눈금 · 실선) vs 육식 합계(오른쪽 눈금 · 점선)
        head = tk.Frame(right, bg=BG)
        head.pack(fill='x', pady=(8, 2))
        tk.Label(head, text='개체수 변화', font=(FONT, 11, 'bold'), bg=BG,
                 anchor='w').pack(side='left')
        for value, text in reversed(GRAPH_RANGES):
            tk.Radiobutton(head, text=text, value=value, variable=self.chart_range,
                           indicatoron=0, takefocus=0, font=(FONT, 9), padx=5,
                           command=self._render_chart).pack(side='right')
        # 격자 높이에서 위 · 아래 글자 영역을 뺀 만큼을 그래프가 쓴다
        chart_h = max(260, w.rows * self.cell - 280)
        self.chart = tk.Canvas(right, width=PANEL_W, height=chart_h, bg='white',
                               highlightthickness=1, highlightbackground='#ced4da')
        self.chart.pack()

        # 선택한 개체 정보
        tk.Label(right, text='선택한 개체 (격자 클릭)', font=(FONT, 11, 'bold'),
                 bg=BG, anchor='w').pack(fill='x', pady=(8, 0))
        self.sel_label = tk.Label(right, text='', font=(FONT, 10), bg=BG,
                                  anchor='w', justify='left', wraplength=PANEL_W)
        self.sel_label.pack(fill='x')

    def _build_map_legend(self, parent):
        width = self.canvas.winfo_reqwidth()
        lg = tk.Canvas(parent, width=width, height=24, bg=BG, highlightthickness=0)
        lg.pack(fill='x', pady=(4, 0))
        s, row_h = 14, 22
        terrain = [('rect', GRASS_PALETTE[-1], '풀 많음'), ('rect', GRASS_PALETTE[0], '풀 없음'),
                   ('rect', BARE_COLOR, '맨땅'), ('rect', WATER_COLOR, '물')]
        burrow_sp = [m for m in self.herbivores if m in BURROW_USERS]
        if self.has_browse:
            terrain.insert(2, ('browse', None, '관목(사슴 먹이)'))
        if self.has_burrow:
            names = '·'.join(SPECIES_NAME.get(m, m) for m in burrow_sp) or '토끼'
            terrain.append(('rect', BURROW_COLOR, f'은신처(굴, {names}만)'))
        animals = []
        for group, members in (('초식:', self.herbivores), ('육식:', self.carnivores)):
            if members:
                animals.append(('label', None, group))
                animals += [('animal', m, SPECIES_NAME.get(m, m)) for m in members]
        if self.has_burrow:
            animals += [('shelter', (m, 'burrow', BURROW_COLOR),
                         f'굴 안 {SPECIES_NAME.get(m, m)}(숨음·안 잡힘)') for m in burrow_sp]
        sprint_sp = [m for m in self.herbivores if m in SPRINTERS]
        animals += [('sprint', m, f'질주 가능 {SPECIES_NAME.get(m, m)}(→ 진행 방향)')
                    for m in sprint_sp]
        animals.append(('danger', None, '빨간 테두리 = 배고픔/목마름 위험'))

        x, y = 6, 12
        for line in (terrain, animals):
            for kind, arg, text in line:
                # 항목 너비를 대략 계산해 넘치면 줄을 바꾼다
                est = s + 8 + 9 * len(text) + (17 if kind == 'danger' else 0)
                if x > 6 and x + est > width:
                    x, y = 6, y + row_h
                if kind == 'rect':
                    lg.create_rectangle(x, y - s / 2, x + s, y + s / 2, fill=arg,
                                        outline='#868e96')
                elif kind == 'browse':
                    # 왼쪽 절반 = 뜯긴 관목, 오른쪽 절반 = 다 자란 관목 + 점 무늬
                    lg.create_rectangle(x, y - s / 2, x + s / 2, y + s / 2,
                                        fill=BROWSE_PALETTE[0], outline='')
                    lg.create_rectangle(x + s / 2, y - s / 2, x + s, y + s / 2,
                                        fill=BROWSE_PALETTE[-1], outline='')
                    lg.create_rectangle(x, y - s / 2, x + s, y + s / 2, outline='#868e96')
                    for fx, fy in BROWSE_DOTS:
                        lx, ly = x + 1 + fx * (s - 4), y - s / 2 + 1 + fy * (s - 4)
                        lg.create_rectangle(lx, ly, lx + 2, ly + 2, fill=BROWSE_DOT, outline='')
                elif kind == 'sprint':
                    lg.create_image(x + s / 2 + 1, y, image=self._sprite(arg, False, 18))
                    self._draw_sprint_arrow(lg, x + s / 2 + 1, y, 4, 18, tags=())
                    x += 3
                elif kind == 'shelter':
                    m, shelter, bg = arg
                    lg.create_rectangle(x - 1, y - s / 2 - 2, x + s + 3, y + s / 2 + 2, fill=bg,
                                        outline='#868e96')
                    lg.create_image(x + s / 2 + 1, y, image=self._sprite(m, False, 18, shelter))
                    x += 3
                elif kind == 'animal':
                    lg.create_image(x + s / 2 + 1, y, image=self._sprite(arg, False, 18))
                    x += 3      # 18px 스프라이트가 14px 칸보다 넓어 글자를 조금 민다
                elif kind == 'danger':
                    lg.create_image(x + s / 2 - 2, y, image=self._sprite(self.species[0], True, 16))
                    if len(self.species) > 1:
                        lg.create_image(x + s / 2 + 15, y,
                                        image=self._sprite(self.species[-1], True, 16))
                    x += 17
                if kind == 'label':
                    t = lg.create_text(x, y, text=text, anchor='w', font=(FONT, 9, 'bold'),
                                       fill='#495057')
                    x = lg.bbox(t)[2] + 6
                    continue
                t = lg.create_text(x + s + 4, y, text=text, anchor='w', font=(FONT, 9))
                x = lg.bbox(t)[2] + 14
            x += 10      # 지형과 동물 사이 간격
        lg.config(height=y + 12)

    def _create_cells(self):
        """지형 이미지: small(1칸=1픽셀)을 cell 배로 확대해 big 에 복사해 그린다."""
        w = self.sim.world
        self._terrain_small = tk.PhotoImage(master=self.root, width=w.cols, height=w.rows)
        self._terrain_big = tk.PhotoImage(master=self.root, width=w.cols * self.cell,
                                          height=w.rows * self.cell)
        self.canvas.create_image(0, 0, image=self._terrain_big, anchor='nw', tags=('terrain',))
        # 지형 무늬(관목 점) — 배경 투명 이미지 1장. 새 배치 때만 다시 만든다.
        self._pattern_img = None
        self._pattern_item = self.canvas.create_image(0, 0, anchor='nw', tags=('pattern',))
        self._cell_keys = [None] * (w.rows * w.cols)
        self._meadow_cells: list = []
        self._browse_cells: list = []
        self._terrain_dirty = True

    # ------------------------------------------------------------ 키 바인딩
    def _bind_keys(self):
        top = self.root.winfo_toplevel()
        top.bind('<space>', lambda e: self.toggle())
        top.bind('<Right>', lambda e: self.step_once())
        top.bind('<r>', lambda e: self.reset())
        top.bind('<R>', lambda e: self.reset())
        top.bind('<g>', lambda e: self.open_graph())
        top.bind('<G>', lambda e: self.open_graph())

    # ------------------------------------------------------------ 그래프 창
    def open_graph(self):
        """개체수 그래프 창을 연다(이미 열려 있으면 앞으로 가져온다)."""
        if self.graph_win is not None and self.graph_win.alive():
            self.graph_win.lift()
            return self.graph_win
        self.graph_win = GraphWindow(self)
        return self.graph_win

    def _graph_closed(self, win):
        if self.graph_win is win:
            self.graph_win = None

    # ------------------------------------------------------------ 조작
    def toggle(self):
        if self.playing:
            self.stop()
        else:
            self.start()

    def start(self):
        if not self.sim.world.animals:
            self.message.config(text='모든 동물이 사라졌습니다. 새 배치(R)로 다시 시작하세요.')
            return
        self.playing = True
        self.play_btn.config(text='⏸ 일시정지 (Space)')
        self._step_times.clear()
        self._schedule(1)

    def stop(self):
        self.playing = False
        if self._timer is not None:
            try:
                self.root.after_cancel(self._timer)
            except tk.TclError:
                pass
            self._timer = None
        try:
            self.play_btn.config(text='▶ 재생 (Space)')
        except tk.TclError:
            pass

    def _schedule(self, delay):
        self._timer = self.root.after(max(1, int(delay)), self._tick)

    def _on_speed(self):
        self._step_times.clear()

    def _tick(self):
        self._timer = None
        if not self.playing:
            return
        t0 = time.perf_counter()
        sp = self.speed.get()
        if sp == 0:
            # 최대 속도: 화면 그리기(프레임당 수십 ms)가 병목이라, 한 프레임에 MAX_SPEED_BUDGET_MS 동안
            # 시뮬레이션만 여러 step 돌리고 한 번 그린다.
            while self.step_once(render=False) and \
                    (time.perf_counter() - t0) * 1000 < MAX_SPEED_BUDGET_MS:
                pass
            self.render()
        else:
            self.step_once()
        if not self.playing:
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        delay = 1 if sp == 0 else 1000 / sp - elapsed_ms
        self._schedule(delay)

    def step_once(self, render=True):
        """시뮬레이션을 1 step 진행하고 (render=True면) 다시 그린다. 진행했으면 True."""
        if not self.sim.world.animals:
            self.stop()
            self.render()
            return False
        try:
            self.sim.step()
        except Exception:
            self.stop()
            traceback.print_exc()
            self.message.config(text='시뮬레이션 오류가 발생했습니다(터미널 로그 확인).')
            return False
        now = time.perf_counter()
        self._step_times.append(now)
        if len(self._step_times) > 600:     # 최대 속도(한 프레임에 여러 step)에서도 몇 초 평균이 되게
            del self._step_times[:100]
        if not self.sim.world.animals:
            self.stop()
            render = True
        if render:
            self.render()
        return self.playing or render

    def reset(self):
        self.stop()
        self.sim.reset(seed=None)
        self._selected_id = None
        self._click_cell = None
        self._world_ref = None      # 같은 World 객체를 재사용해도 지형을 다시 읽도록
        self.message.config(text='')
        self.render()

    def _on_click(self, event):
        w = self.sim.world
        r, c = int(event.y // self.cell), int(event.x // self.cell)
        if not (0 <= r < w.rows and 0 <= c < w.cols):
            return
        here = [a for a in w.animals if a.row == r and a.col == c]
        if not here:
            self._selected_id = None
            self._click_cell = (r, c)
        else:
            ids = [a.id for a in here]
            if self._selected_id in ids:       # 같은 칸을 다시 누르면 다음 개체
                nxt = ids[(ids.index(self._selected_id) + 1) % len(ids)]
            else:
                nxt = ids[0]
            self._selected_id = nxt
            self._click_cell = None
        self.render()

    _click_cell = None

    # ------------------------------------------------------------ 그리기
    def _refresh_terrain_cache(self):
        """World 객체가 바뀌면(새 배치) 지형 정보를 다시 읽는다."""
        w = self.sim.world
        if self._world_ref is w and getattr(self, '_terrain_turn', -1) <= w.turn:
            self._terrain_turn = w.turn
            return
        self._world_ref = w
        self._terrain_turn = w.turn
        burrow = getattr(w, 'is_burrow', None)
        if not callable(burrow):
            burrow = None
        browse = w.is_browse if self.has_browse else None
        terrain = []
        for r in range(w.rows):
            for c in range(w.cols):
                if w.is_water(r, c):
                    terrain.append('W')
                elif burrow is not None and burrow(r, c):
                    terrain.append('U')
                elif browse is not None and browse(r, c):
                    terrain.append('V')
                elif w.is_meadow(r, c):
                    terrain.append('M')
                else:
                    terrain.append('B')
        self._meadow_cells = [(i, divmod(i, w.cols)) for i, t in enumerate(terrain) if t == 'M']
        self._browse_cells = [(i, divmod(i, w.cols)) for i, t in enumerate(terrain) if t == 'V']
        for i, t in enumerate(terrain):     # 고정 지형 칸 색(풀 · 관목은 매 프레임 단계로 칠함)
            self._cell_keys[i] = t if t not in ('M', 'V') else None
        self._terrain_dirty = True
        self._build_pattern([rc for _, rc in self._browse_cells])

    def _build_pattern(self, browse_cells):
        """관목 칸에 점 무늬(줄기)를 찍은 투명 이미지를 지형 위에 겹친다(칸이 없으면 비움)."""
        if not browse_cells:
            self._pattern_img = None
            self.canvas.itemconfig(self._pattern_item, image='')
            return
        w, s = self.sim.world, self.cell
        img = tk.PhotoImage(master=self.root, width=w.cols * s, height=w.rows * s)
        d = max(1, round(s / 8))
        for r, c in browse_cells:
            x0, y0 = c * s, r * s
            for fx, fy in BROWSE_DOTS:
                lx, ly = x0 + int(fx * (s - d)), y0 + int(fy * (s - d))
                img.put(BROWSE_DOT, to=(lx, ly, lx + d, ly + d))
        self._pattern_img = img
        self.canvas.itemconfig(self._pattern_item, image=img)

    def _render_cells(self):
        """풀 · 관목 자란 정도를 단계로 나눠, 단계가 바뀐 칸만 다시 칠한다."""
        w = self.sim.world
        keys = self._cell_keys
        changed = []
        layers = [(self._meadow_cells, w.grass_level, GRASS_STEPS, 0)]
        if self._browse_cells:
            layers.append((self._browse_cells, w.browse_level, BROWSE_STEPS, BROWSE_KEY0))
        for cells, level, steps, base in layers:
            for i, (r, c) in cells:
                k = int(level(r, c) * steps + 0.5)
                if k < 0:
                    k = 0
                elif k > steps:
                    k = steps
                k += base
                if keys[i] != k:
                    keys[i] = k
                    changed.append(i)
        cols = w.cols
        if self._terrain_dirty or len(changed) > 300:
            # 바뀐 칸이 많으면 작은 이미지 전체를 한 번에 넣는 편이 빠르다
            colors = [_key_color(k) for k in keys]
            rows = ['{' + ' '.join(colors[r * cols:(r + 1) * cols]) + '}' for r in range(w.rows)]
            self._terrain_small.put(' '.join(rows), to=(0, 0))
        else:
            put = self._terrain_small.put
            for i in changed:
                r, c = divmod(i, cols)
                put(_key_color(keys[i]), to=(c, r))
        if self._terrain_dirty or changed:
            self._terrain_big.tk.call(self._terrain_big, 'copy', self._terrain_small,
                                      '-zoom', self.cell, self.cell)
        self._terrain_dirty = False

    def _sprite(self, species, danger, size, shelter=None):
        key = (species, bool(danger), int(size), shelter)
        img = self._sprites.get(key)
        if img is None:
            img = self._sprites[key] = make_sprite(self.root, species, int(size), bool(danger),
                                                   shelter)
        return img

    def _shelter_of(self, a):
        """개체가 지금 쓰고 있는 은신처: 'burrow' | None (지형 캐시 기준)."""
        k = self._cell_keys[a.row * self.sim.world.cols + a.col] \
            if 0 <= a.row < self.sim.world.rows and 0 <= a.col < self.sim.world.cols else None
        kind = _SHELTER_KEY.get(k) if isinstance(k, str) else None
        if kind == 'burrow' and a.species in BURROW_USERS:
            return kind
        return None

    @staticmethod
    def is_sprinting(a) -> bool:
        """질주 조건을 채웠는가: 같은 방향으로 sprint_streak번 이상 연속 이동했다(다음에 같은 방향이면 2칸 갈 수 있음)."""
        need = SPRINT_STREAK.get(a.species)
        if need is None:
            return False
        return (getattr(a, 'streak', 0) or 0) >= need and (getattr(a, 'last_dir', 0) or 0) in (1, 2, 3, 4)

    @staticmethod
    def _draw_sprint_arrow(cv, cx, cy, direction, cell, tags=('sprint',)):
        """칸 가장자리에 진행 방향을 가리키는 작은 삼각형(질주 표식)을 그린다."""
        dr, dc = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        if not (dr or dc):
            return
        tip, base, half = cell * 0.62, cell * 0.28, max(2.5, cell * 0.2)
        tx, ty = cx + dc * tip, cy + dr * tip
        bx, by = cx + dc * base, cy + dr * base
        px, py = -dr * half, dc * half          # 진행 방향에 수직
        cv.create_polygon(tx, ty, bx + px, by + py, bx - px, by - py, fill=SPRINT_COLOR,
                          outline=SPRINT_EDGE, width=1, tags=tags)

    def _render_animals(self, selected):
        cv = self.canvas
        c = self.cell
        cv.delete('animal')
        cv.delete('sprint')
        groups: dict[tuple[int, int], list] = {}
        for a in self.sim.world.animals:
            groups.setdefault((a.row, a.col), []).append(a)
        half = c / 2
        create = cv.create_image
        sprite = self._sprite
        shelter_of = self._shelter_of
        counts: dict[tuple[str, str], int] = {}      # (은신처, 종) → 마리 수
        sprinting = []
        check_sprint = bool(SPRINT_STREAK)
        for (r, col), grp in groups.items():
            n = len(grp)
            for k, a in enumerate(grp):
                dx = 0 if n == 1 else (k - (n - 1) / 2) * c * 0.35
                danger = a.energy < DANGER or a.hydration < DANGER
                shelter = shelter_of(a)
                if shelter:
                    counts[shelter, a.species] = counts.get((shelter, a.species), 0) + 1
                x, y = col * c + half + dx, r * c + half
                create(x, y, image=sprite(a.species, danger, c, shelter), tags=('animal',))
                if check_sprint and self.is_sprinting(a):
                    sprinting.append((a, x, y))
        # 질주 표식은 모든 동물 위에(옆 칸 동물에 가리지 않게)
        for a, x, y in sprinting:
            self._draw_sprint_arrow(cv, x, y, a.last_dir, c)
        self.shelter_counts = counts
        self.sprinting_ids = {a.id for a, _, _ in sprinting}
        if selected is not None:
            x0, y0 = selected.col * c, selected.row * c
            cv.create_rectangle(x0 - 2, y0 - 2, x0 + c + 2, y0 + c + 2, outline=SELECT_COLOR,
                                width=3, tags=('animal',))

    def _render_vision(self, selected):
        cv = self.canvas
        cv.delete('vision')
        if selected is None:
            if self._click_cell is not None:
                r, c = self._click_cell
                s = self.cell
                cv.create_rectangle(c * s, r * s, (c + 1) * s, (r + 1) * s,
                                    outline=SELECT_COLOR, width=2, tags=('vision',))
            return
        s = self.cell
        inside = self._diamond(selected, VISION.get(selected.species, 4))
        for r, c in inside:
            cv.create_rectangle(c * s, r * s, (c + 1) * s, (r + 1) * s, fill=SELECT_COLOR,
                                stipple='gray25', outline='', tags=('vision',))
        # 시야 경계선(맨해튼 다이아몬드 계단 모양, 주황 실선)
        self._draw_outline(inside, '#e67700', 2, None)
        # 포식자: 사냥 반경 경계선(빨간 점선) — 이동 후 이 안의 먹이를 잡으려 한다
        if is_carnivore(selected.species):
            hunt = self._diamond(selected, HUNT_RANGE.get(selected.species, 1))
            self._draw_outline(hunt, HUNT_RANGE_COLOR, 3, (4, 2))

    def _diamond(self, a, radius):
        """개체 a 에서 맨해튼 거리 radius 이내인 격자 안 칸 집합."""
        w = self.sim.world
        r0, c0 = a.row, a.col
        inside = set()
        for dr in range(-radius, radius + 1):
            span = radius - abs(dr)
            for dc in range(-span, span + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < w.rows and 0 <= c < w.cols:
                    inside.add((r, c))
        return inside

    def _draw_outline(self, inside, color, width, dash):
        """칸 집합의 바깥 경계선만 그린다."""
        cv, s = self.canvas, self.cell
        kw = dict(fill=color, width=width, tags=('vision',))
        if dash:
            kw['dash'] = dash
        for r, c in inside:
            x0, y0, x1, y1 = c * s, r * s, (c + 1) * s, (r + 1) * s
            if (r - 1, c) not in inside:
                cv.create_line(x0, y0, x1, y0, **kw)
            if (r + 1, c) not in inside:
                cv.create_line(x0, y1, x1, y1, **kw)
            if (r, c - 1) not in inside:
                cv.create_line(x0, y0, x0, y1, **kw)
            if (r, c + 1) not in inside:
                cv.create_line(x1, y0, x1, y1, **kw)

    def _render_panel(self, st):
        w = self.sim.world
        self.header.config(text=f'Turn {w.turn}   ·   {self._policy_label()}')
        sp = self.speed.get()
        real = ''
        if self.playing and len(self._step_times) >= 2:
            dt = self._step_times[-1] - self._step_times[0]
            if dt > 0:
                real = f' (실제 {(len(self._step_times) - 1) / dt:.1f})'
        state = '재생 중' if self.playing else '정지'
        self.subheader.config(
            text=f'{state} · 설정 속도 {"최대" if sp == 0 else sp} step/s{real}'
                 f' · 프레임 {self.last_render_ms:.1f} ms · 동물 {len(w.animals)}마리')

        self.count_label.config(text=self._count_text(st))
        self.env_label.config(text=self._env_text(st))

        total = sum(int(st.get(k, 0) or 0) for k, _ in CAUSES)
        parts = [f'{label} {int(st.get(k, 0) or 0)}' for k, label in CAUSES]
        pred_parts = [f'{SPECIES_NAME.get(s, s)} {int(st[f"deaths_predation_{s}"])}'
                      for s in self.species if f'deaths_predation_{s}' in st]
        if pred_parts:
            parts[-1] += f'({" · ".join(pred_parts)})'
        self.deaths_label.config(text='누적 사망: ' + ' · '.join(parts) + f' · 합계 {total}')

    def _count_text(self, st):
        """종별 마릿수 두 줄: 초식 합계 — 종별 / 육식 합계 — 종별. 0마리면 (멸종)."""
        lines = []
        for title, members in (('초식', self.herbivores), ('육식', self.carnivores)):
            if not members:
                continue
            counts = [int(st.get(f'count_{s}', 0) or 0) for s in members]
            each = ' · '.join(f'{SPECIES_NAME.get(s, s)} {n}' + (' (멸종)' if n == 0 else '')
                              for s, n in zip(members, counts))
            lines.append(f'{title} 합계 {sum(counts)}마리 — {each}')
        return '\n'.join(lines)

    def _env_text(self, st):
        """먹이층 비율 · 굴 안 개체 수 · 질주 / 2칸 사냥 누적(있는 키만)."""
        food = []
        if 'grass_ratio' in st:
            food.append(f'풀 {float(st["grass_ratio"] or 0) * 100:.0f}%')
        if 'browse_ratio' in st:
            food.append(f'관목(사슴 먹이) {float(st["browse_ratio"] or 0) * 100:.0f}%')
        lines = []
        if food:
            lines.append('먹이층(자란 정도): ' + ' · '.join(food))
        extra = []
        if self.has_burrow:
            for s in self.herbivores:
                if s in BURROW_USERS:
                    extra.append(f'굴 안 {SPECIES_NAME.get(s, s)} '
                                 f'{self.shelter_counts.get(("burrow", s), 0)}마리')
        for s in self.herbivores:
            if s in SPRINTERS:
                n_now = sum(1 for a in self.sim.world.animals if a.id in self.sprinting_ids
                            and a.species == s)
                cum = st.get(f'sprints_{s}')
                extra.append(f'질주 가능 {SPECIES_NAME.get(s, s)} {n_now}마리'
                             + (f'(누적 {int(cum)})' if cum is not None else ''))
        for s in self.carnivores:
            far = st.get(f'far_hunts_{s}')
            if far is not None:
                extra.append(f'{SPECIES_NAME.get(s, s)} 2칸 사냥 누적 {int(far)}')
        imm = [f'{SPECIES_NAME.get(s, s)} {int(st[f"immigrants_{s}"])}'
               for s in self.carnivores if f'immigrants_{s}' in st]
        if imm:
            extra.append('외부 유입 누적 ' + ' · '.join(imm))
        if extra:
            lines.append(' · '.join(extra))
        return '\n'.join(lines)

    def _render_chart(self):
        """패널 그래프: 초식 합계(왼쪽 눈금 · 실선) vs 육식 합계(오른쪽 눈금 · 점선), 범위는 chart_range."""
        cv = self.chart
        cv.delete('all')
        rng = self.chart_range.get()
        a, b, t_min = _range_window(self._series, rng)
        W, H = int(cv['width']), int(cv['height'])
        self.chart_info = draw_dual_chart(
            cv, self._series, (0, 0, W, H), self._total_line(HERB_TOTAL),
            self._total_line(CARN_TOTAL), a, b, t_min, smooth=_smooth_for(rng), big=False)

    def _total_line(self, key):
        """(키, 이름, 색) — 초식 / 육식 합계 선."""
        members = self.herbivores if key == HERB_TOTAL else self.carnivores
        kind = '초식' if key == HERB_TOTAL else '육식'
        names = '+'.join(SPECIES_NAME.get(s, s) for s in members)
        return key, f'{kind} 합계({names})', kind, TOTAL_COLOR[key]

    def _render_selection_text(self, selected):
        if selected is not None:
            a = selected
            warn = []
            if a.energy < DANGER:
                warn.append('배고픔 위험')
            if a.hydration < DANGER:
                warn.append('목마름 위험')
            kind = '육식' if is_carnivore(a.species) else '초식'
            shelter = self._shelter_of(a)
            where = {'burrow': '  · 굴 안(포식자에게 안 보이고 안 잡힘)'}.get(shelter, '')
            reach = f'시야 반경 {VISION.get(a.species, "?")}칸(맨해튼, 노란 영역 · 주황 테두리)'
            if is_carnivore(a.species):
                reach += f'\n사냥 반경 {HUNT_RANGE.get(a.species, 1)}칸(빨간 점선)'
            elif a.species in SPRINT_STREAK:
                need = SPRINT_STREAK[a.species]
                streak = int(getattr(a, 'streak', 0) or 0)
                state = '질주 가능(같은 방향이면 2칸)' if self.is_sprinting(a) else '질주 불가'
                reach += f'\n같은 방향 연속 이동 {streak}/{need} → {state}'
            self.sel_label.config(
                text=f'#{a.id} {SPECIES_NAME.get(a.species, a.species)}({kind})'
                     f'  위치 ({a.row}, {a.col})  나이 {a.age}{where}\n'
                     f'energy {a.energy:.1f} (배고픔 {100 - a.energy:.0f})   '
                     f'hydration {a.hydration:.1f} (목마름 {100 - a.hydration:.0f})\n'
                     + reach + (f'\n⚠ {", ".join(warn)}' if warn else ''),
                fg='#c92a2a' if warn else '#212529')
        elif self._selected_id is not None:
            self.sel_label.config(text=f'#{self._selected_id} 개체는 사망했습니다.', fg='#868e96')
        elif self._click_cell is not None:
            r, c = self._click_cell
            w = self.sim.world
            burrow = getattr(w, 'is_burrow', None)
            b_names = '·'.join(SPECIES_NAME.get(s, s) for s in BURROW_USERS)
            if w.is_water(r, c):
                kind = '물웅덩이'
            elif callable(burrow) and burrow(r, c):
                kind = (f'은신처(굴) — {b_names}만 들어갈 수 있고, 안에 있으면 '
                        f'포식자에게 보이지 않고 잡히지 않음. 풀·물 없음')
            elif self.has_browse and w.is_browse(r, c):
                kind = (f'관목(사슴 먹이) · 자란 정도 {w.browse_level(r, c) * 100:.0f}% — '
                        f'사슴만 먹고 토끼는 먹지 않음. 누구나 지나갈 수 있음')
            elif w.is_meadow(r, c):
                kind = f'초지 · 풀 {w.grass_level(r, c) * 100:.0f}%'
            else:
                kind = '맨땅(풀이 자라지 않음)'
            self.sel_label.config(text=f'칸 ({r}, {c}): {kind}', fg='#212529')
        else:
            self.sel_label.config(text='동물을 클릭하면 상태와 시야 범위(포식자는 사냥 반경도)를 보여 줍니다.',
                                  fg='#868e96')

    def render(self):
        """현재 상태 전체를 그린다(셀은 바뀐 칸만)."""
        t0 = time.perf_counter()
        w = self.sim.world
        self._refresh_terrain_cache()
        self._render_cells()
        selected = None
        if self._selected_id is not None:
            for a in w.animals:
                if a.id == self._selected_id:
                    selected = a
                    break
        self._render_vision(selected)
        self._render_animals(selected)
        st = w.stats()
        self._render_panel(st)
        self._series.sync(w)
        self._render_chart()
        self._render_selection_text(selected)
        if not w.animals:
            self.message.config(text='모든 동물이 사라졌습니다. 자동 정지 — 새 배치(R)로 다시 시작하세요.')
        self.last_render_ms = (time.perf_counter() - t0) * 1000
        if self.graph_win is not None:
            # 재생 중에는 시간 간격(GRAPH_INTERVAL)마다만, 정지 상태(한 step · 새 배치)에서는 바로 다시 그린다
            self.graph_win.request(force=not self.playing)

    # ------------------------------------------------------------ 기타
    def _policy_label(self):
        mode = getattr(self.sim, 'policy_mode', '?')
        return POLICY_NAME.get(mode, str(mode))

    def destroy(self):
        self.stop()
        if self.graph_win is not None:
            self.graph_win.close()
        try:
            self.root.winfo_toplevel().destroy()
        except tk.TclError:
            pass


# ---------------------------------------------------------------- 시계열 캐시
HERB_TOTAL = 'herb_total'     # 초식 합계 열 이름(SeriesCache 파생 열)
CARN_TOTAL = 'carn_total'     # 육식 합계 열 이름
TOTAL_COLOR = {HERB_TOTAL: '#2a78d6', CARN_TOTAL: '#d9480f'}   # 초식 = 파랑(차가운 색), 육식 = 주홍(따뜻한 색)


class SeriesCache:
    """world.history 를 열별 숫자 목록 + 누적합으로 들고 있는 캐시.

    매 프레임 새로 생긴 history 원소만 읽는다(16,000 step이 되어도 프레임당 비용은 그대로).
    누적합 덕분에 구간 평균 · 이동 평균을 O(1)에 구한다. derived = {새 열: [더할 열들]}로
    합계 열(초식 합계 등)을 함께 만든다. World 가 바뀌거나(새 배치) history 가 짧아지면 처음부터 다시 만든다.
    """

    def __init__(self, keys, derived=None):
        self.base_keys = tuple(keys)
        self.derived = {k: tuple(v) for k, v in (derived or {}).items()}
        self.keys = self.base_keys + tuple(self.derived)
        self._reset(None)

    def _reset(self, world):
        self.world = world
        self.first = None
        self.n = 0
        self.turn: list = []
        self.values = {k: [] for k in self.keys}
        self.prefix = {k: [0.0] for k in self.keys}
        self.present: set = set()

    def sync(self, world):
        h = getattr(world, 'history', None) or []
        if (world is not self.world or len(h) < self.n
                or (self.n and h and h[0] is not self.first)):
            self._reset(world)
        if not h or len(h) == self.n:
            return
        if self.first is None:
            self.first = h[0]
            self.present = {k for k in self.base_keys if k in h[0]}
            self.present |= {k for k, parts in self.derived.items()
                             if parts and all(p in self.present for p in parts)}
        turn, vals, pre = self.turn, self.values, self.prefix
        base, derived = self.base_keys, self.derived.items()
        for i in range(self.n, len(h)):
            row = h[i]
            turn.append(row.get('turn', i))
            got = {}
            for k in base:
                try:
                    v = float(row.get(k) or 0.0)
                except (TypeError, ValueError):
                    v = 0.0
                got[k] = v
                vals[k].append(v)
                pre[k].append(pre[k][-1] + v)
            for k, parts in derived:
                v = sum(got[p] for p in parts)
                vals[k].append(v)
                pre[k].append(pre[k][-1] + v)
        self.n = len(h)

    def has(self, key):
        return key in self.present

    def mean(self, key, a, b):
        """[a, b) 구간 평균."""
        if b <= a:
            return self.values[key][a] if 0 <= a < self.n else 0.0
        p = self.prefix[key]
        return (p[b] - p[a]) / (b - a)

    def smooth(self, key, i, half):
        """i 를 가운데로 한 이동 평균(창 = 2·half + 1, 양 끝은 있는 값만)."""
        return self.mean(key, max(0, i - half), min(self.n, i + half + 1))


# ---------------------------------------------------------------- 두 축 시계열 그리기
GRAPH_INTERVAL = 0.3          # 재생 중 그래프 창 다시 그리기 최소 간격(초)
GRAPH_SMOOTH = 101            # 이동 평균 창(step) — plots.py 의 LV 차트 · analysis.SMOOTH_WINDOW 와 같음
GRAPH_RANGES = (('all', '전체'), ('2000', '최근 2,000'), ('300', '최근 300'))


def _smooth_for(rng):
    """범위별 이동 평균 창: 최근 300 step은 101 step 평균이면 너무 뭉개져서 15 step."""
    return 15 if rng == '300' else GRAPH_SMOOTH


def _range_window(ser, rng):
    """범위('all' | '2000' | '300') → history 색인 구간 [a, b)와 가로축 최소 길이(step)."""
    if rng == 'all' or not str(rng).isdigit():
        return 0, ser.n, 1
    k = int(rng)
    return max(0, ser.n - k), ser.n, k


def _lighten(color: str, f: float = 0.72) -> str:
    """색을 흰색 쪽으로 f 만큼 섞는다(캔버스에는 투명도가 없어 '흐린 선'을 이렇게 만든다)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return _rgb(round(v + (255 - v) * f) for v in (r, g, b))


def _nice_step(v: float) -> float:
    """v 이상인 1, 2, 5 × 10^k 중 가장 작은 값(눈금 간격)."""
    if v <= 0:
        return 1
    mag = 1.0
    while mag * 10 <= v:
        mag *= 10
    while mag > v:
        mag /= 10
    for m in (1, 2, 5, 10):
        if m * mag >= v:
            return m * mag
    return 10 * mag


def _peaks(ys, half, prominence):
    """표본 목록의 봉우리 위치: 앞뒤 half 개 안에서 최댓값이고 그 창의 최솟값보다 prominence 이상 높다."""
    out = []
    n = len(ys)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        if b - a < half + 2:          # 가장자리(창이 반도 안 됨)는 아직 봉우리인지 모른다
            continue
        win = ys[a:b]
        v = ys[i]
        if v >= max(win) and v - min(win) >= prominence:
            if not out or i - out[-1] > half:
                out.append(i)
    return out


def draw_dual_chart(cv, ser, box, left, right, a, b, t_min=1, smooth=GRAPH_SMOOTH, big=False,
                    title=None):
    """캔버스 box 안에 두 축 시계열을 그린다(lv_timeseries 차트와 같은 구성).

    left / right = (열 이름, 범례 이름, 축 이름, 색). 왼쪽 = 실선 · 왼쪽 눈금, 오른쪽 = 점선 · 오른쪽 눈금.
    진한 선 = smooth step 이동 평균, 흐린 띠 = 원값(가로 1픽셀 구간의 최소~최대). 두 축은 따로 자동 눈금.
    가로축 = history 색인 [a, b) (최소 t_min step 폭). big=True 면 큰 글자 · 봉우리 · 한 줄 요약.
    돌려주는 dict: summary(한 줄 요약), ymax(왼쪽, 오른쪽), peaks({열: [turn, ...]}).
    """
    X0, Y0, X1, Y1 = box
    fs = 10 if big else 8
    info = {'summary': '', 'ymax': (0, 0), 'peaks': {}}
    lk, lname, laxis, lcol = left
    rk, rname, raxis, rcol = right
    x0, x1 = X0 + (64 if big else 38), X1 - (64 if big else 38)
    top = Y0 + (12 if big else 4)
    # 제목(있으면) + 범례 한 줄
    lx = x0 if title is None else X0 + 12
    ly = top + 9
    if title:
        t = cv.create_text(lx, ly, text=title, anchor='w', font=(FONT, fs + 2, 'bold'))
        lx = cv.bbox(t)[2] + 18
    for color, dash, text in ((lcol, None, f'{lname} · 실선'), (rcol, (6, 3), f'{rname} · 점선')):
        kw = {'dash': dash} if dash else {}
        cv.create_line(lx, ly, lx + 22, ly, fill=color, width=3, **kw)
        t = cv.create_text(lx + 27, ly, text=text, anchor='w', font=(FONT, fs + 1))
        lx = cv.bbox(t)[2] + 14
    if big:
        cv.create_polygon(lx, ly + 5, lx + 5, ly - 5, lx + 10, ly + 5, fill=lcol, outline='')
        cv.create_polygon(lx + 14, ly - 5, lx + 19, ly + 5, lx + 24, ly - 5, fill=rcol, outline='')
        cv.create_text(lx + 30, ly, text='봉우리', anchor='w', font=(FONT, fs + 1))
    # 축 제목: 각 눈금 위에 가로로(Windows Tk 는 회전한 한글을 깨뜨려 그린다)
    ay = top + (40 if big else 30)
    cv.create_text(X0 + 6, ay, text=f'{laxis} 수(마리) · 왼쪽 눈금', anchor='w',
                   font=(FONT, fs, 'bold'), fill=lcol)
    cv.create_text(X1 - 6, ay, text=f'{raxis} 수(마리) · 오른쪽 눈금', anchor='e',
                   font=(FONT, fs, 'bold'), fill=rcol)
    y0 = ay + (16 if big else 13)
    y1 = Y1 - (48 if big else 22)
    if x1 - x0 < 40 or y1 - y0 < 30:
        return info
    cv.create_rectangle(x0, y0, x1, y1, outline='#adb5bd')
    n = b - a
    if n <= 0 or not (ser.has(lk) or ser.has(rk)):
        cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, text='데이터 없음', font=(FONT, fs + 1),
                       fill='#868e96')
        return info
    turns = ser.turn
    t_start, t_last = turns[a], turns[b - 1]
    t_end = max(t_last, t_start + t_min - 1, t_start + 1)
    sx = (x1 - x0) / (t_end - t_start)

    def X(t):
        return x0 + (t - t_start) * sx

    # 가로 1픽셀 구간: 원값 최소~최대 + 구간 가운데의 이동 평균
    nb = max(1, min(n, int((x1 - x0) * (t_last - t_start + 1) / (t_end - t_start + 1))))
    edges = [a + n * j // nb for j in range(nb + 1)]
    spans = [(e0, e1) for e0, e1 in zip(edges, edges[1:]) if e1 > e0]
    tx = [turns[(e0 + e1 - 1) // 2] for e0, e1 in spans]
    half = smooth // 2
    series = {}
    for key in (lk, rk):
        vals = ser.values[key]
        env = [(min(vals[e0:e1]), max(vals[e0:e1])) for e0, e1 in spans]
        sm = [ser.smooth(key, (e0 + e1 - 1) // 2, half) for e0, e1 in spans]
        series[key] = (env, sm)

    def axis_max(key):
        top_v = max([1.0] + [hi for _, hi in series[key][0]])
        step = _nice_step(top_v / 4)
        return step * -(-top_v // step), step

    (ym_l, st_l), (ym_r, st_r) = axis_max(lk), axis_max(rk)
    info['ymax'] = (ym_l, ym_r)

    def Yl(v):
        return y1 - (y1 - y0) * v / ym_l

    def Yr(v):
        return y1 - (y1 - y0) * v / ym_r

    # 눈금
    v = 0.0
    while v <= ym_l + 1e-9:
        y = Yl(v)
        if 0 < v < ym_l:
            cv.create_line(x0 + 1, y, x1 - 1, y, fill='#eef0f2')
        cv.create_text(x0 - 5, y, text=f'{v:g}', anchor='e', font=(FONT, fs), fill=lcol)
        v += st_l
    v = 0.0
    while v <= ym_r + 1e-9:
        cv.create_text(x1 + 5, Yr(v), text=f'{v:g}', anchor='w', font=(FONT, fs), fill=rcol)
        v += st_r
    xs_step = _nice_step((t_end - t_start) / (7 if big else 4))
    t = -(-t_start // xs_step) * xs_step
    while t <= t_end:
        x = X(t)
        cv.create_line(x, y1, x, y1 + 4, fill='#868e96')
        cv.create_text(x, y1 + 5, text=f'{int(t)}', anchor='n', font=(FONT, fs))
        t += xs_step
    cv.create_text(x1 + (36 if big else 34), y1 + 5, text='turn', anchor='ne',
                   font=(FONT, fs), fill='#868e96')

    # 선: 흐린 원값 띠 → 진한 이동 평균(오른쪽 = 점선)
    for key, Y, color, dash in ((lk, Yl, lcol, None), (rk, Yr, rcol, (6, 3))):
        env, sm = series[key]
        band = []
        for x, (lo, hi) in zip(tx, env):
            band += [X(x), Y(lo), X(x), Y(hi)]
        if len(band) >= 4:
            cv.create_line(*band, fill=_lighten(color), width=1)
        if len(sm) >= 2:
            pts = []
            for x, v in zip(tx, sm):
                pts += [X(x), Y(v)]
            kw = {'dash': dash} if dash else {}
            cv.create_line(*pts, fill=color, width=2, **kw)
        elif sm:
            cv.create_oval(X(tx[0]) - 2, Y(sm[0]) - 2, X(tx[0]) + 2, Y(sm[0]) + 2, fill=color,
                           outline='')
    if not big:
        return info

    # 봉우리(이동 평균 기준): 왼쪽 ▲, 오른쪽 ▼ — 앞뒤 약 400 step 안에서 가장 높고, 범위의 25% 이상 솟은 점
    if t_last - t_start >= 1200 and len(spans) > 4:
        per_px = (t_last - t_start) / len(spans)
        hw = max(2, int(400 / per_px))
        for key, Y, color, up in ((lk, Yl, lcol, True), (rk, Yr, rcol, False)):
            sm = series[key][1]
            idx = _peaks(sm, hw, max(0.25 * (max(sm) - min(sm)), 1e-9))
            info['peaks'][key] = [tx[i] for i in idx]
            for i in idx:
                x, y = X(tx[i]), Y(sm[i])
                if up:
                    cv.create_polygon(x - 5, y - 4, x + 5, y - 4, x, y - 13, fill=color,
                                      outline='white')
                else:
                    cv.create_polygon(x - 5, y - 13, x + 5, y - 13, x, y - 4, fill=color,
                                      outline='white')
    # 한 줄 요약(이동 평균 기준, 표시 구간)
    sl, sr = series[lk][1], series[rk][1]
    parts = [f'표시 {t_start}~{t_last} step', f'{laxis} {min(sl):.0f}~{max(sl):.0f}마리',
             f'{raxis} {min(sr):.0f}~{max(sr):.0f}마리 (이동 평균 최소~최대)']
    pl, pr = info['peaks'].get(lk, []), info['peaks'].get(rk, [])
    if len(pl) >= 2:
        gap = (pl[-1] - pl[0]) / (len(pl) - 1)
        parts.append(f'{laxis} 봉우리 {len(pl)}개, 간격 평균 약 {gap:.0f} step')
        lags = [next(q for q in pr if p <= q < p + gap) - p
                for p in pl if any(p <= q < p + gap for q in pr)]
        if lags:
            parts.append(f'{raxis} 봉우리는 평균 약 {sum(lags) / len(lags):.0f} step 뒤')
    info['summary'] = ' · '.join(parts)
    cv.create_text(x0, y1 + 26, text=info['summary'], anchor='nw', font=(FONT, fs - 1),
                   fill='#495057', width=x1 - x0)
    return info


# ---------------------------------------------------------------- 개체수 그래프 창
GRAPH_MODES = (('total', '초식 합계 vs 육식 합계'), ('pairs', '종 쌍별(포식자–먹이)'))


class GraphWindow:
    """개체수 그래프 창(Toplevel). 기본 = 초식 합계 vs 육식 합계 한 장, 전환 = 포식자–먹이 쌍별(위아래 칸).

    쌍은 config.PREY 에서 각 포식 종의 첫 번째 먹이(여우–토끼, 늑대–사슴). 가로축은 step 0부터 지금까지
    (또는 최근 2,000 / 300 step). 표준 tkinter 캔버스에 직접 그린다(matplotlib 없음).
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.top = tk.Toplevel(viewer.root)
        self.top.title(f'개체수 그래프 · {viewer._policy_label()}')
        self.top.configure(bg=BG)
        self.top.minsize(640, 420)
        self.top.protocol('WM_DELETE_WINDOW', self.close)
        sp = viewer.species
        self.pairs = []
        for c in viewer.carnivores:
            prey = next((p for p in PREY.get(c, ()) if p in sp), None)
            if prey is not None:
                self.pairs.append((c, prey))
        self.range = tk.StringVar(master=self.top, value='all')
        self.mode = tk.StringVar(master=self.top, value='total')
        self.last_draw_ms = 0.0
        self.draws = 0
        self.summaries: list = []
        self._last_time = 0.0
        self._pending = None

        bar = tk.Frame(self.top, bg=BG)
        bar.pack(fill='x', padx=10, pady=(8, 2))
        self.title_label = tk.Label(bar, text='', font=(FONT, 11, 'bold'), bg=BG, anchor='w')
        self.title_label.pack(side='left')
        for value, text in reversed(GRAPH_RANGES):
            tk.Radiobutton(bar, text=text, value=value, variable=self.range, indicatoron=0,
                           takefocus=0, font=(FONT, 10), padx=8,
                           command=lambda: self.request(force=True)).pack(side='right')
        tk.Label(bar, text='   범위:', font=(FONT, 10), bg=BG).pack(side='right', padx=(0, 4))
        for value, text in reversed(GRAPH_MODES):
            tk.Radiobutton(bar, text=text, value=value, variable=self.mode, indicatoron=0,
                           takefocus=0, font=(FONT, 10), padx=8,
                           command=lambda: self.request(force=True)).pack(side='right')
        tk.Label(bar, text='보기:', font=(FONT, 10), bg=BG).pack(side='right', padx=(0, 4))
        self.canvas = tk.Canvas(self.top, width=1180, height=700, bg='white',
                                highlightthickness=1, highlightbackground='#ced4da')
        self.canvas.pack(fill='both', expand=True, padx=10, pady=(2, 10))
        self.canvas.bind('<Configure>', lambda e: self._debounce())
        for key, fn in (('<space>', viewer.toggle), ('<Right>', viewer.step_once),
                        ('<r>', viewer.reset), ('<R>', viewer.reset)):
            self.top.bind(key, lambda e, f=fn: f())
        self.request(force=True)

    # ---------------------------------------------------------- 수명
    def alive(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def lift(self):
        try:
            self.top.deiconify()
            self.top.lift()
        except tk.TclError:
            pass

    def close(self):
        if self._pending is not None:
            try:
                self.top.after_cancel(self._pending)
            except tk.TclError:
                pass
            self._pending = None
        try:
            self.top.destroy()
        except tk.TclError:
            pass
        self.viewer._graph_closed(self)

    def _debounce(self):
        """창 크기를 바꾸는 동안에는 잠깐 기다렸다가 한 번만 다시 그린다."""
        if self._pending is not None:
            try:
                self.top.after_cancel(self._pending)
            except tk.TclError:
                pass
        self._pending = self.top.after(60, self._resized)

    def _resized(self):
        self._pending = None
        self.request(force=True)

    def request(self, force=False):
        """다시 그리기 요청. 재생 중에는 GRAPH_INTERVAL 초에 한 번만 실제로 그린다."""
        if not self.alive():
            self.viewer._graph_closed(self)
            return
        now = time.perf_counter()
        if not force and now - self._last_time < GRAPH_INTERVAL:
            return
        self.viewer._series.sync(self.viewer.sim.world)
        self.draw()
        self._last_time = time.perf_counter()

    # ---------------------------------------------------------- 그리기
    def draw(self):
        t0 = time.perf_counter()
        cv = self.canvas
        cv.delete('all')
        v = self.viewer
        ser = v._series
        # 창 크기를 따른다(아직 화면에 안 나왔으면 요청 크기)
        W = cv.winfo_width() if cv.winfo_width() > 1 else int(cv['width'])
        H = cv.winfo_height() if cv.winfo_height() > 1 else int(cv['height'])
        rng = self.range.get()
        smooth = _smooth_for(rng)
        a, b, t_min = _range_window(ser, rng)
        self.title_label.config(
            text=f'Turn {v.sim.world.turn} · {v._policy_label()} · 진한 선 = {smooth} step 이동 평균,'
                 f' 흐린 띠 = 원값')
        if self.mode.get() == 'pairs' and self.pairs:
            panels = []
            for pred, prey in self.pairs:
                lc, rc = v.line_color[prey], v.line_color[pred]
                np_, nq = SPECIES_NAME.get(prey, prey), SPECIES_NAME.get(pred, pred)
                panels.append((f'{nq}–{np_}', (f'count_{prey}', f'{np_}(먹이)', np_, lc),
                               (f'count_{pred}', f'{nq}(포식자)', nq, rc)))
        else:
            panels = [('초식 vs 육식', v._total_line(HERB_TOTAL), v._total_line(CARN_TOTAL))]
        self.summaries = []
        ph = H / len(panels)
        for k, (title, left, right) in enumerate(panels):
            info = draw_dual_chart(cv, ser, (0, k * ph, W, (k + 1) * ph), left, right, a, b,
                                   t_min, smooth=smooth, big=True, title=title)
            self.summaries.append(info['summary'])
        self.draws += 1
        self.last_draw_ms = (time.perf_counter() - t0) * 1000


def _nice_ceil(v: float) -> float:
    """그래프 축 최댓값을 보기 좋은 수(1, 2, 5 × 10^k)로 올린다."""
    if v <= 0:
        return 1
    mag = 1
    while mag * 10 <= v:
        mag *= 10
    for m in (1, 2, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def run(sim) -> None:
    """뷰어 창을 띄우고 tkinter mainloop 를 실행한다."""
    root = tk.Tk()
    viewer = Viewer(root, sim)
    root.resizable(False, False)
    root.protocol('WM_DELETE_WINDOW', viewer.destroy)
    root.mainloop()
