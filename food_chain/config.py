"""먹이사슬 생태계(토끼 · 사슴 · 여우 · 늑대) 설정값.

모든 수치는 이 파일 한 곳에서 관리한다. World는 생성 시점에 이 모듈의 값을 읽는다.
수치의 현실 근거는 docs/수치_근거.md 참고(종 간 '비율의 방향'을 맞추는 것이 목표).
"""
from pathlib import Path

# ── 격자 ────────────────────────────────────────────────────────────────
ROWS = 50
COLS = 75

# ── 종 ─────────────────────────────────────────────────────────────────
SPECIES = ("rabbit", "deer", "fox", "wolf")
HERBIVORES = ("rabbit", "deer")
CARNIVORES = ("fox", "wolf")
SPECIES_LABEL = {"rabbit": "토끼", "deer": "사슴", "fox": "여우", "wolf": "늑대"}
# 초식 = SARSA, 육식 = Q-Learning
ALGORITHM = {"rabbit": "sarsa", "deer": "sarsa", "fox": "qlearning", "wolf": "qlearning"}

# 먹이 관계: 포식 종 → 잡을 수 있는 종. (늑대가 여우를 공격하는 관계는 없다.)
PREY = {"fox": ("rabbit",), "wolf": ("deer", "rabbit")}
# 피식 종 → 자신을 잡을 수 있는 종
PREDATORS = {h: tuple(c for c in CARNIVORES if h in PREY[c]) for h in HERBIVORES}
# 먹이 크기(늑대 State의 '먹이 크기' 성분): 1 = 소형, 2 = 대형
PREY_SIZE = {"rabbit": 1, "deer": 2}

# 시야 반경(맨해튼 거리). 포식자는 자기가 잡는 먹이 종보다 최소 1칸 넓어야 한다(포식–피식 쌍 기준).
# 사슴 5는 사용자 지정값(늑대의 사냥 반경이 2칸이 된 만큼 한 칸 더 멀리 본다).
VISION = {"rabbit": 4, "deer": 5, "fox": 6, "wolf": 8}
# State의 거리 성분은 시야와 무관하게 2구간(1 = 가까움, 2 = 멂)만 쓴다: 시야를 넓혀도 표 크기는 그대로다.
# NEAR_DIST[종] = '가까움'으로 보는 최대 거리.
#   육식: 자기 사냥 반경(hunt_range)에 맞춘다 → '가까움' = 지금 사냥을 시도할 수 있는 거리.
#         (여우는 인접 사냥이지만 한 걸음이면 닿는 2칸까지를 가까움으로 본다.)
#   초식: 자신을 잡는 포식자의 사냥 반경 + 1 → '가까움' = 다음 step에 사정권에 들 수 있는 거리.
#         토끼는 여우(반경 1) 기준 2칸, 사슴은 늑대(반경 2) 기준 3칸(시야 5 중 1~3 가까움, 4~5 멂).
NEAR_DIST = {"rabbit": 2, "deer": 3, "fox": 2, "wolf": 2}

MAX_ENERGY = 100.0
MAX_HYDRATION = 100.0
FAMINE_BELOW = 35.0           # 이 값 미만이면 '기근 대사'(종별 famine_cost_mult)로 버틴다

# 행동: 0 정지, 1 위, 2 아래, 3 왼쪽, 4 오른쪽 (방향 코드와 같은 번호를 쓴다)
ACTIONS = ("정지", "위", "아래", "왼쪽", "오른쪽")
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

# ── 지형 ────────────────────────────────────────────────────────────────
MEADOW_COVERAGE = 0.45        # 전체 칸 중 풀이 자랄 수 있는 칸의 목표 비율
WATER_MAX_DIST = 20           # 어느 칸에서든 물가까지의 최단 이동 거리 상한
POND_GRID = (2, 3)            # 물웅덩이를 (행, 열) 구역마다 1개씩 배치
POND_RADIUS_ROWS = (2.6, 4.0)  # 웅덩이(타원) 세로 반지름 범위
POND_RADIUS_COLS = (4.5, 7.0)  # 웅덩이(타원) 가로 반지름 범위
# 굴(burrow): BURROW_USERS만 들어갈 수 있는 은신처. 굴 안의 동물은 잡히지 않고 포식자에게 보이지 않는다.
# 굴에는 풀도 물도 없으므로 계속 숨어 있을 수는 없다. 총 칸 수가 '먹이 바닥 개체수'를 정한다.
BURROW_USERS = ("rabbit",)
BURROW_FRACTION = 0.025       # 땅 칸 중 굴 칸의 비율(0이면 굴 없음). 사용자 지정값(0.05에서 절반으로)
BURROW_CLUSTER = 4            # 굴 군집 하나의 칸 수(2×2)
# (사슴용 덤불은 없앴다. 사슴의 방어 수단은 은신이 아니라 질주다: PARAMS["deer"]의 sprint_* 참고.)
GRASS_REGROW_STEPS = 80       # 뜯긴 풀이 0 → 1로 다시 자라는 데 걸리는 step
GRASS_EDIBLE_LEVEL = 0.5      # 이 수준 이상 자란 풀만 먹을 수 있다
# 관목(browse): 사슴 전용 먹이층. 현실의 노루는 관목의 잎 · 새순을 골라 먹는 browser이고
# 토끼는 풀을 뜯는 grazer라서 먹이가 겹치지 않는다(niche partitioning).
# 관목 칸은 풀밭 · 물 · 굴이 아닌 땅에 군집으로 놓이고, 풀보다 느리게 다시 자란다(목본은 생장이 느리다).
# BROWSE_ENABLED = False면 관목을 만들지 않고 사슴도 예전처럼 풀을 먹는다(토끼와 경쟁).
BROWSE_ENABLED = True
DIET = {"rabbit": "grass", "deer": "browse"}   # 초식 종별 먹이층
BROWSE_COVERAGE = 0.20        # 전체 칸 중 관목이 자랄 수 있는 칸의 목표 비율
BROWSE_REGROW_STEPS = 120     # 뜯긴 관목이 0 → 1로 다시 자라는 데 걸리는 step
BROWSE_EDIBLE_LEVEL = 0.5     # 이 수준 이상 자란 관목만 먹을 수 있다
DRINK_AMOUNT = 30.0           # 물가에 있으면 step마다 회복하는 hydration
YIELD_ABOVE_HYDRATION = 70.0  # 물가의 동물이 이 값 이상이면, 목마른(<35) 같은 식성 동물에게 자리를 바꿔 준다

# ── 종별 생리 수치 ──────────────────────────────────────────────────────
# 대사(move_cost): energy는 0~100 '비축량'이므로 step당 감소는 체중당 대사율(∝ 체중^-0.25)에 해당한다.
#   체중 토끼 2 / 여우 6 / 사슴 26 / 늑대 45 kg → 상대값 1.0 / 0.76 / 0.53 / 0.46
#   → move_cost 토끼 1.0 > 여우 0.8 > 사슴 0.6 > 늑대 0.5 (정지 = 절반). 큰 동물일수록 오래 버틴다.
# 갈증(thirst): step당 감소 = move_cost. 쉬면 energy 소모는 절반이 되지만 수분은 그대로 줄어
#   물 없이 버티는 시간이 먹이 없이 버티는 시간보다 짧다(현실: 탈수가 아사보다 빠르다).
#   대신 먹이에 든 수분이 조금 들어온다(풀: grass_hydration, 사냥: HUNT의 hydration).
# 공통: initial_count, move_cost(이동 1회 energy), stay_cost(정지 1회), thirst(step당 hydration 감소),
#       max_age(개체별 수명 = max_age × U(0.8, 1.2)), breed_prob(조건 충족 시 step당 번식 확률),
#       breed_min_energy, breed_min_hydration, breed_cooldown, population_cap(안전장치)
#       breed_cost(에너지 예산 번식: 번식할 때 부모가 내는 energy = 새끼의 시작 energy.
#       newborn_energy를 따로 주면 새끼는 그 값으로 시작한다. ENERGY_BUDGET_BREEDING 참고)
#       maturity(새끼가 번식할 수 있게 되기까지의 step. 없으면 breed_cooldown)
# 초식: grass_energy(다 자란 풀 한 칸의 energy), grass_hydration(다 자란 풀 한 칸의 수분),
#       graze_extra(자기 칸 외에 함께 뜯는 인접 칸 수),
#       eat_below(energy가 이 값 미만일 때만 풀을 뜯는다)
#       (관목을 먹는 종(사슴)에게는 grass_* · graze_extra의 '풀'이 관목을 뜻한다.)
#       sprint_prob(질주: 같은 방향으로 sprint_streak번 이상 연속 이동한 뒤 또 같은 방향을 고르면
#       이 확률로 한 번에 2칸 이동. 둘째 칸이 막혀 있으면 1칸만. 없으면 질주하지 않는다),
#       sprint_streak(질주에 필요한 연속 이동 횟수), sprint_cost(질주로 더 간 1칸의 추가 energy)
# 육식: hunt_below(energy가 이 값 미만일 때만 사냥한다 = 포만 규칙)
#       hunt_range(사냥 반경: 이동 후 이 맨해튼 거리 안의 먹이를 포획 시도. 없으면 1 = 인접만)
#       territory_limit(영역성: 번식 시도가 (1 − 개체수/territory_limit) 확률로만 성사된다. 없으면 제한 없음)
#       forage_gain(잡식: energy가 FAMINE_BELOW 미만이고 초원 칸에 있으면 step마다 얻는 energy. 없으면 0)
#       famine_cost_mult(기근 내성: energy가 FAMINE_BELOW 미만이면 이동 · 정지 비용에 곱한다. 없으면 1)
PARAMS = {
    "rabbit": {
        "initial_count": 120,
        "move_cost": 1.0,
        "stay_cost": 0.5,
        "thirst": 1.0,
        "max_age": 400,
        "breed_prob": 0.02,
        "breed_min_energy": 60.0,
        "breed_min_hydration": 40.0,
        "breed_cooldown": 40,
        "breed_cost": 30.0,
        "population_cap": 700,
        "grass_energy": 16.0,
        "grass_hydration": 8.0,
        "graze_extra": 0,
        "eat_below": 90.0,
    },
    "deer": {
        "initial_count": 60,
        "move_cost": 0.6,
        "stay_cost": 0.3,
        "thirst": 0.6,
        "max_age": 1200,
        "breed_prob": 0.015,       # 0.01 → 0.015: 늑대에게 눌린 뒤 회복이 늑대 수명(1000)보다 늦지 않게
        "breed_min_energy": 70.0,
        "breed_min_hydration": 40.0,
        "breed_cooldown": 300,
        "breed_cost": 35.0,
        "population_cap": 400,
        "grass_energy": 12.0,      # 관목 한 칸의 energy(잎 · 새순은 풀보다 영양가가 높다)
        "grass_hydration": 5.0,
        "graze_extra": 1,          # 한 번에 두 칸(자기 칸 + 인접 1칸)을 뜯는다
        "eat_below": 85.0,
        "sprint_prob": 0.5,        # 직선 질주(사슴은 직선 주로에서 늑대보다 빠르다)
        "sprint_streak": 2,
        "sprint_cost": 0.3,        # 질주 1회 = 이동 0.6 + 0.3 (2칸을 따로 걷는 1.2보다는 싸다)
    },
    "fox": {
        # 8 → 20: 처음부터 평형 근처에서 시작한다(8이면 토끼가 430까지 불었다가 여우가 76까지 폭증한 뒤 함께 무너졌다)
        "initial_count": 20,
        "move_cost": 0.8,
        "stay_cost": 0.4,
        "thirst": 0.8,
        "max_age": 800,
        "breed_prob": 0.03,
        # 90 → 80, 쿨다운 250 → 100: 번식 횟수를 쿨다운이 아니라 먹은 양(energy)이 정하게 한다.
        # 사냥은 energy < 40(hunt_below)일 때만 하므로, 토끼를 잡아 80을 넘겨야만 새끼를 낼 수 있다.
        "breed_min_energy": 80.0,
        "breed_min_hydration": 40.0,
        "breed_cooldown": 100,
        "breed_cost": 40.0,
        "population_cap": 200,     # 80 → 200: 영역성 상한을 없앴으므로 안전장치로만 둔다
        # 70 → 40: 정말 배고플 때만 사냥한다. 비축 energy가 작아져 토끼가 줄면 곧바로(약 50~100 step)
        # 굶어 죽는다. 70일 때는 여우가 비축분으로 100 step 넘게 버티며 남은 토끼를 바닥냈다.
        "hunt_below": 40.0,
        "hunt_range": 1,           # 여우는 단독 매복형: 인접한 먹이만
        # 0.5 → 1.0(절감 없음): 굶주린 여우가 오래 버티면 토끼가 줄어든 뒤에도 계속 잡아 저점이 깊어진다
        "famine_cost_mult": 1.0,
        # 영역성 상한(40)은 없앴다: 에너지 예산 번식에서는 토끼가 줄면 번식이 저절로 멈추고
        # 새끼부터 굶어 죽으므로, 여우 수가 토끼 수에 끌려간다.
        "territory_limit": None,
        "forage_gain": 0.0,        # 잡식 먹이: 0.4로 켜면 여우가 저점에서도 줄지 않아 토끼가 멸종했다(기본 꺼짐)
    },
    "wolf": {
        "initial_count": 6,
        "move_cost": 0.5,
        "stay_cost": 0.25,
        "thirst": 0.5,
        "max_age": 1000,
        "breed_prob": 0.02,
        # 90 → 80: 사냥은 energy < 70일 때만 하므로 토끼(+5)로는 80에 못 닿고 사슴(+45)을 먹어야만
        # 번식한다(늑대 수가 사슴 수를 뒤따른다). 90일 때는 번식 가능 구간이 너무 짧아 노화를 못 따라갔다.
        "breed_min_energy": 80.0,
        "breed_min_hydration": 40.0,
        "breed_cooldown": 300,     # 150 → 300: 연 1회 번식(사슴과 같은 주기). 짧으면 사슴을 바닥낸다
        "breed_cost": 40.0,
        "population_cap": 40,
        "hunt_below": 70.0,
        "hunt_range": 2,           # 늑대는 무리 추격형: 2칸 안의 먹이까지 사냥 시도
        # 0.25 → 0.7: 대사를 현실 비율(0.5)로 낮춘 만큼 기근 절감도 현실 범위(10~30%)로 완화
        "famine_cost_mult": 0.7,
        "territory_limit": 14,     # 24 → 14: 관목 분리 뒤 사슴 상한이 약 130으로 낮아졌다. 24면 늑대가 17마리까지 불어 사슴을 30까지 누른 뒤 굶어 멸종했다
    },
}

# 에너지 예산 번식(NetLogo Wolf-Sheep · Strannegård 2026과 같은 계열. NetLogo는 부모 energy를 반으로
# 나누고, 여기서는 문턱 + 고정 비용을 쓴다): 번식하면 부모가 breed_cost만큼의
# energy를 새끼에게 떼어 준다. 먹이를 많이 먹어 energy가 남을 때만 잇달아 번식할 수 있고,
# 먹이가 줄면 breed_min_energy에 못 미쳐 번식이 멈춘다. 4종 모두 같은 규칙이다.
# False면 예전 방식(부모 energy를 새끼와 반씩 나눔)으로 돌아간다.
ENERGY_BUDGET_BREEDING = True

# ── 사냥 ────────────────────────────────────────────────────────────────
# (포식 종, 피식 종)별 수치
#   pounce   : 배고픈 포식자가 먹이 칸으로 진입을 시도할 때 포획 확률(실패하면 제자리)
#   catch    : 이동 후 먹이가 상하좌우 인접일 때 포획 확률(진입을 시도한 step에는 적용 안 함)
#   catch_far: 이동 후 먹이가 2칸 이상 ~ hunt_range 안일 때 포획 확률(hunt_range ≥ 2인 포식자만)
#   energy   : 포획 시 energy 증가,  hydration: 포획 시 hydration 증가(먹이의 수분)
#   reward   : 사냥 보상 base (× 욕구 계수)
HUNT = {
    # 여우→토끼: 1.0 / 0.5 → 0.24 / 0.12. 토끼는 여우를 떨쳐 내지 못해 추격이 잡힐 때까지 이어지므로
    # step당 확률은 '한 마리를 잡는 데 드는 시간(≈ 1/catch step)'을 정한다. 높으면 토끼가 드물어져도
    # 여우가 굶지 않아 토끼를 바닥냈고(0.5: 800 step에 토끼 멸종), 0.05면 토끼가 320마리여도 여우가 굶어 죽었다.
    ("fox", "rabbit"): {"pounce": 0.24, "catch": 0.12, "energy": 50.0, "hydration": 20.0,
                        "reward": 2.0},
    # 늑대→토끼: 2칸 추격은 통하지 않고(catch_far 0), 얻는 energy도 적다(사슴의 1/9).
    # 토끼는 굶주림을 늦추는 보조 먹이일 뿐, 토끼만으로는 번식(energy ≥ 80)할 수 없다.
    ("wolf", "rabbit"): {"pounce": 0.1, "catch": 0.05, "catch_far": 0.0, "energy": 5.0,
                         "hydration": 5.0, "reward": 1.0},
    # 늑대→사슴: 2칸 거리 포획은 인접의 절반(0.2로 같게 두면 늑대 6~10마리가 사슴을 멸종시켰다).
    # energy 90 → 45: 늑대 대사를 절반(0.96 → 0.5)으로 낮추면서 먹이 값도 절반으로 낮췄다
    # (한 마리가 차지하는 몫). 그대로 두면 늑대 1마리가 필요한 사슴 수가 반으로 줄어
    # 영역성 상한(24)에 걸린 늑대가 사슴을 누르지 못하고 사슴이 풀 한계(약 330)까지 불어났다.
    ("wolf", "deer"): {"pounce": 0.4, "catch": 0.2, "catch_far": 0.1, "energy": 45.0,
                       "hydration": 30.0, "reward": 3.0},
}

# ── 외부 유입(rescue effect) ─────────────────────────────────────────────
# 포식자 종이 below마리 미만이면 step마다 prob 확률로 맵 가장자리에 1마리가 들어온다(eval 모드만).
# 처음부터 0마리로 시작한 종(대조 실험)에는 적용하지 않는다. 끄려면 IMMIGRATION_ENABLED = False.
IMMIGRATION_ENABLED = True
IMMIGRATION = {
    "fox": {"below": 2, "prob": 0.01},
    "wolf": {"below": 3, "prob": 0.01},
}

# ── 보상 ────────────────────────────────────────────────────────────────
REWARD_GRASS = 0.5            # 풀 먹기 base (× 욕구 계수)
REWARD_WATER = 0.3            # 물 마시기 base (× 욕구 계수)
REWARD_SURVIVE = 0.01         # 살아서 한 step을 넘길 때마다
REWARD_DEATH = -5.0           # 아사 · 탈수 · 포식
REWARD_AGE_DEATH = 0.0        # 노화
REWARD_BLOCKED = -0.05        # 경계 · 물로 막힌 이동
SHAPE_FLEE = 0.05             # 초식: 가까운(NEAR_DIST 이내) 위협과의 거리를 늘림
SHAPE_APPROACH_DANGER = -0.05 # 초식: 가까운 위협과의 거리를 줄임
SHAPE_CHASE = 0.05            # 육식: 먹이와의 거리를 줄임
SHAPE_LOSE_PREY = -0.02       # 육식: 먹이와의 거리를 늘림
SHAPE_WATER = 0.1             # 공통: 목마를 때(hydration < 35) 물가에 가까워지면 +, 멀어지면 −

# ── 학습 ────────────────────────────────────────────────────────────────
ALPHA = 0.1
ALPHA_END = 0.02              # phase 끝의 학습률(ALPHA와 같으면 고정)
ALPHA_SAMPLE_AVERAGE = True   # True면 α = max(ALPHA, 1/방문횟수): 처음 몇 번은 표본 평균
GAMMA = 0.95
EPSILON_START = 0.3
EPSILON_END = 0.05
EVAL_EPSILON = 0.05           # 평가 · 관람 시 탐험 확률
TRAIN_EPISODES = 200
TRAIN_STEPS = 400             # episode 길이(도달해도 terminal이 아님)
TRAIN_COUNT_RANGE = (0.5, 1.5)    # episode마다 초기 개체수 배율
TRAIN_REGROW_RANGE = (0.7, 1.6)   # episode마다 풀 재생 시간 배율
# train 모드(번식 OFF · 재생성) 개체수: 학습 경험이 충분하도록 종별로 따로 정한다.
# 육식은 평가 때 수가 적으므로 더 많이 두고, 초식은 풀이 부족하지 않게 평가 초기값 근처로 둔다.
TRAIN_COUNT = {"rabbit": 160, "deer": 60, "fox": 30, "wolf": 14}

# ── 평가 ────────────────────────────────────────────────────────────────
EVAL_STEPS = 4000             # 진동 주기가 길어 길게 본다
EVAL_SEEDS = 6
BENCH_STEPS = 400

# ── 경로 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
Q_TABLE_DIR = ROOT / "data" / "q_tables"
RESULTS_DIR = ROOT / "data" / "results"

assert set(HERBIVORES) | set(CARNIVORES) == set(SPECIES)
assert all(VISION[c] >= VISION[p] + 1 for c in CARNIVORES for p in PREY[c]), \
    "포식자의 시야는 자기가 잡는 먹이 종보다 최소 1칸 넓어야 합니다."
assert all(NEAR_DIST[s] < VISION[s] for s in SPECIES)
assert all((c, p) in HUNT for c in CARNIVORES for p in PREY[c])
assert all(PARAMS[c].get("hunt_range", 1) <= VISION[c] for c in CARNIVORES)
assert all(PARAMS[s].get("breed_cost", 0.0) < PARAMS[s]["breed_min_energy"] for s in SPECIES)
assert set(DIET) == set(HERBIVORES) and set(DIET.values()) <= {"grass", "browse"}
# 관목을 먹는 종(BROWSE_ENABLED가 꺼져 있으면 없음 = 모두 풀을 먹는다)
BROWSERS = tuple(s for s in HERBIVORES if BROWSE_ENABLED and DIET[s] == "browse")
# 질주하는 종(sprint_prob가 있는 초식)
SPRINTERS = tuple(s for s in HERBIVORES if PARAMS[s].get("sprint_prob"))
# 사냥 반경이 2칸 이상인 포식 종
RANGED_HUNTERS = tuple(c for c in CARNIVORES if PARAMS[c].get("hunt_range", 1) > 1)
