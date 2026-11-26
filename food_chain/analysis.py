"""로트카-볼테라(LV) 유사성 측정: 포식–피식 쌍의 주기 · 포식자 지연 · 진폭 · 포식 사망 비율.

표준 라이브러리만 사용한다. 개체수 시계열을 이동 평균으로 평활하고(SMOOTH_WINDOW),
초기 과도 구간(BURN_IN)을 버린 뒤 다음을 잰다.
    (1) 주기        : 피식자 개체수(선형 추세 제거)의 자기상관이 처음 음수가 된 뒤 나오는 첫 양의 봉우리
    (2) 포식자 지연 : 교차상관 corr(피식자[t], 포식자[t + lag])가 최대인 lag(±주기/2 안에서 탐색).
                      LV 이론은 주기의 약 0.25배(포식자가 1/4 주기 늦게 따라간다).
    (3) 진폭        : 평활한 개체수의 최대/최소 비
    (4) 주기 수     : 분석 구간 길이 / 주기 (봉우리 개수도 함께 적는다)
    (5) 교차상관 최대값
    (6) 포식 사망 비율: 피식자가 줄어드는 구간에서 피식자 사망 중 그 포식자에게 잡힌 비율
주기를 찾지 못했거나 관측된 주기가 MIN_CYCLES회 미만이면 '측정 불가'(measurable = 0)로 둔다.
"""
import csv
from pathlib import Path

from . import config

# 포식–피식 쌍 (포식자, 피식자)
PAIRS = (("fox", "rabbit"), ("wolf", "deer"))
LV_STEPS = 8000            # --lv 기본 step 수
LV_SEEDS = 4               # --lv 기본 seed 수
BURN_IN = 1000             # 초기 과도 구간(분석에서 제외)
SMOOTH_WINDOW = 101        # 이동 평균 창(step, 홀수)
SAMPLE_EVERY = 4           # 상관 계산용 솎아내기 간격(step)
ACF_MIN_PEAK = 0.15        # 자기상관 봉우리로 인정하는 최소 값
MIN_CYCLES = 2.0           # 이보다 적게 관측되면 '측정 불가'
MIN_LENGTH = 1000          # 분석 구간(과도 구간 제외, 멸종 전)이 이보다 짧으면 '측정 불가'

SUMMARY_COLUMNS = (
    "policy", "seed", "predator", "prey", "steps", "extinct_step", "analyzed_steps",
    "measurable", "period", "period_predator", "period_peaks", "cycles", "peaks",
    "lag", "lag_ratio", "xcorr_max", "amplitude_prey", "amplitude_predator",
    "min_prey", "max_prey", "min_predator", "max_predator",
    "predation_share", "decline_steps")


# ── 신호 처리 ────────────────────────────────────────────────────────────
def moving_average(x, window):
    """가운데 맞춤 이동 평균(양 끝은 있는 값만으로 평균). 길이는 그대로."""
    n = len(x)
    if window <= 1 or n == 0:
        return [float(v) for v in x]
    half = window // 2
    prefix = [0.0]
    for v in x:
        prefix.append(prefix[-1] + v)
    out = []
    for i in range(n):
        lo = i - half if i > half else 0
        hi = i + half + 1 if i + half + 1 < n else n
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def detrend(x):
    """최소제곱 직선을 뺀다(평균도 0이 된다)."""
    n = len(x)
    if n < 2:
        return [0.0] * n
    mt = (n - 1) / 2.0
    mx = sum(x) / n
    sxx = sum((i - mt) ** 2 for i in range(n))
    slope = sum((i - mt) * (v - mx) for i, v in enumerate(x)) / sxx
    return [v - mx - slope * (i - mt) for i, v in enumerate(x)]


def autocorrelation(x, max_lag):
    """평균을 뺀 x의 자기상관 r[0..max_lag] (r[0] = 1). 분산이 0이면 전부 0."""
    n = len(x)
    m = sum(x) / n if n else 0.0
    d = [v - m for v in x]
    var = sum(v * v for v in d)
    if var <= 0.0:
        return [0.0] * (max_lag + 1)
    out = []
    for lag in range(max_lag + 1):
        out.append(sum(d[i] * d[i + lag] for i in range(n - lag)) / var)
    return out


def first_acf_peak(acf, min_peak=ACF_MIN_PEAK):
    """자기상관이 처음 음수가 된 뒤 나오는 첫 양의 구간에서 가장 높은 점의 lag. 없으면 None."""
    n = len(acf)
    i = 1
    while i < n and acf[i] > 0.0:
        i += 1
    while i < n:
        while i < n and acf[i] <= 0.0:
            i += 1
        if i >= n:
            return None
        j = i
        while j < n and acf[j] > 0.0:
            j += 1
        if j >= n:                      # 양의 구간이 탐색 범위 끝에서 잘렸다: 봉우리를 확정할 수 없다
            return None
        k = max(range(i, j), key=acf.__getitem__)
        if acf[k] >= min_peak:
            return k
        i = j
    return None


def correlation(x, y):
    """피어슨 상관계수. 길이가 2 미만이거나 분산이 0이면 0."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    sxy = sxx = syy = 0.0
    for i in range(n):
        a = x[i] - mx
        b = y[i] - my
        sxy += a * b
        sxx += a * a
        syy += b * b
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def best_lag(prey, predator, max_lag):
    """corr(prey[t], predator[t + lag])가 최대인 lag(−max_lag ~ +max_lag)와 그 값.

    lag > 0이면 포식자가 피식자를 뒤따른다.
    """
    n = min(len(prey), len(predator))
    best, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if n - abs(lag) < 3:
            continue
        if lag >= 0:
            r = correlation(prey[:n - lag], predator[lag:n])
        else:
            r = correlation(prey[-lag:n], predator[:n + lag])
        if r > best_r:
            best, best_r = lag, r
    return best, best_r


def count_peaks(x, half_width):
    """앞뒤 half_width 안에서 최대이고 평균보다 큰 점(봉우리)의 위치 목록(양 끝 half_width 구간 제외)."""
    n = len(x)
    if n == 0 or half_width < 1:
        return []
    mean = sum(x) / n
    peaks = []
    i = half_width
    while i < n - half_width:
        v = x[i]
        if v > mean and v >= max(x[i - half_width:i + half_width + 1]):
            peaks.append(i)
            i += half_width
        else:
            i += 1
    return peaks


# ── 쌍 분석 ──────────────────────────────────────────────────────────────
def analyze_pair(prey, predator, prey_deaths=None, prey_killed=None, burn_in=BURN_IN,
                 window=SMOOTH_WINDOW, sample_every=SAMPLE_EVERY):
    """한 쌍의 step별 시계열을 분석해 dict를 돌려준다.

    prey, predator : step별 개체수
    prey_deaths    : step별 피식자 사망 수(전체 원인),  prey_killed: 그중 이 포식자에게 잡힌 수
    둘 중 한 종이 0마리가 되면 그 직전까지만 분석한다(extinct_step에 그 step을 적는다).
    """
    n = min(len(prey), len(predator))
    extinct = -1
    for i in range(n):
        if prey[i] <= 0 or predator[i] <= 0:
            extinct = i
            break
    end = n if extinct < 0 else extinct
    out = {"steps": n - 1, "extinct_step": extinct, "analyzed_steps": max(0, end - burn_in),
           "measurable": 0, "period": None, "period_predator": None, "period_peaks": None,
           "cycles": None, "peaks": None, "lag": None, "lag_ratio": None, "xcorr_max": None,
           "amplitude_prey": None, "amplitude_predator": None,
           "min_prey": None, "max_prey": None, "min_predator": None, "max_predator": None,
           "predation_share": None, "decline_steps": None}
    if end - burn_in < max(MIN_LENGTH, 4 * window):
        return out
    sp = moving_average(prey[:end], window)[burn_in:]
    sq = moving_average(predator[:end], window)[burn_in:]
    length = len(sp)
    out["min_prey"], out["max_prey"] = min(sp), max(sp)
    out["min_predator"], out["max_predator"] = min(sq), max(sq)
    out["amplitude_prey"] = max(sp) / min(sp) if min(sp) > 0 else None
    out["amplitude_predator"] = max(sq) / min(sq) if min(sq) > 0 else None

    # 포식 사망 비율: 평활한 피식자 수가 줄어드는 step에서의 (포식 사망 / 전체 사망)
    if prey_deaths is not None and prey_killed is not None:
        dead = killed = steps = 0
        for i in range(1, length):
            if sp[i] < sp[i - 1]:
                t = burn_in + i
                dead += prey_deaths[t]
                killed += prey_killed[t]
                steps += 1
        out["decline_steps"] = steps
        out["predation_share"] = killed / dead if dead else None

    # 상관은 솎아낸 시계열로 계산한다(평활 창이 간격보다 훨씬 넓어 정보 손실이 없다)
    dp = detrend(sp[::sample_every])
    dq = detrend(sq[::sample_every])
    m = len(dp)
    acf = autocorrelation(dp, m // 2)
    k = first_acf_peak(acf)
    kq = first_acf_peak(autocorrelation(dq, m // 2))
    if kq is not None:
        out["period_predator"] = kq * sample_every
    max_lag = k // 2 if k is not None else min(m // 4, 1000 // sample_every)
    lag, r = best_lag(dp, dq, max(1, max_lag))
    out["lag"] = lag * sample_every
    out["xcorr_max"] = r
    if k is None:
        return out
    period = k * sample_every
    out["period"] = period
    out["cycles"] = length / period
    out["lag_ratio"] = out["lag"] / period
    peaks = count_peaks(sp, max(1, int(period * 0.35)))
    out["peaks"] = len(peaks)
    if len(peaks) >= 2:
        out["period_peaks"] = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
    out["measurable"] = int(out["cycles"] >= MIN_CYCLES)
    return out


# ── 실행 · 저장 · 출력 ───────────────────────────────────────────────────
def series_keys():
    """seed별 시계열 CSV(lv_<policy>.csv)의 열(seed 다음부터)."""
    keys = ["turn"] + [f"count_{s}" for s in config.SPECIES]
    keys += [f"deaths_{s}" for s in config.HERBIVORES]                 # 그 step의 사망(전체 원인)
    keys += [f"kills_{c}_{p}" for c in config.CARNIVORES for p in config.PREY[c]]   # 그 step의 포획
    return keys


def run_series(policy, seed, steps, tables=None):
    """eval 모드로 steps만큼 돌려 step별 기록(list of dict, series_keys() 키)을 돌려준다."""
    from .simulation import Simulation
    sim = Simulation(seed=seed, policy_mode=policy, tables=tables)
    w = sim.world
    w.record_history = False
    pairs = [(c, p) for c in config.CARNIVORES for p in config.PREY[c]]
    last = {cp: 0 for cp in pairs}

    def row():
        r = {"turn": w.turn}
        for s in config.SPECIES:
            r[f"count_{s}"] = w.n_alive[s]
        for s in config.HERBIVORES:
            r[f"deaths_{s}"] = w.deaths[s]
        for c, p in pairs:
            total = w.metrics[f"kills_{c}_{p}"]
            r[f"kills_{c}_{p}"] = total - last[(c, p)]
            last[(c, p)] = total
        return r

    rows = [row()]
    for _ in range(steps):
        sim.step()
        rows.append(row())
        if not w.animals:
            break
    return rows


def analyze_series(rows, **kw):
    """run_series() 결과에서 PAIRS의 각 쌍을 분석한다. {(포식자, 피식자): dict}."""
    out = {}
    for c, p in PAIRS:
        out[(c, p)] = analyze_pair([r[f"count_{p}"] for r in rows],
                                   [r[f"count_{c}"] for r in rows],
                                   [r[f"deaths_{p}"] for r in rows],
                                   [r[f"kills_{c}_{p}"] for r in rows], **kw)
    return out


def _fmt(v, spec=".0f", none="-"):
    return none if v is None else format(v, spec)


def print_lv(rows, out=print):
    """lv_summary 행들을 쌍 · 정책별 표로 출력한다."""
    label = config.SPECIES_LABEL
    groups = {}
    for r in rows:
        groups.setdefault((r["policy"], r["predator"], r["prey"]), []).append(r)
    for (policy, c, p), g in groups.items():
        out(f"  [{policy}] {label[c]}–{label[p]}  (LV 이론: 지연/주기 ≈ 0.25)")
        out(f"    {'seed':>5} {'판정':<6} {'주기':>6} {'봉우리간격':>8} {'주기수':>6} {'지연':>6} "
            f"{'지연/주기':>8} {'교차상관':>7} {'진폭(피식)':>9} {'진폭(포식)':>9} "
            f"{'피식 최소~최대':>14} {'포식 최소~최대':>13} {'포식사망비율':>9} {'멸종 step':>8}")
        for r in g:
            ok = "측정" if r["measurable"] else "측정 불가"
            out(f"    {r['seed']:>5} {ok:<6} {_fmt(r['period']):>6} {_fmt(r['period_peaks']):>8} "
                f"{_fmt(r['cycles'], '.1f'):>6} {_fmt(r['lag']):>6} {_fmt(r['lag_ratio'], '.2f'):>8} "
                f"{_fmt(r['xcorr_max'], '.2f'):>7} {_fmt(r['amplitude_prey'], '.1f'):>9} "
                f"{_fmt(r['amplitude_predator'], '.1f'):>9} "
                f"{_fmt(r['min_prey']):>6}~{_fmt(r['max_prey']):<7} "
                f"{_fmt(r['min_predator'], '.1f'):>6}~{_fmt(r['max_predator'], '.1f'):<6} "
                f"{_fmt(r['predation_share'], '.2f'):>9} "
                f"{(str(r['extinct_step']) if r['extinct_step'] >= 0 else '-'):>8}")
        ok = [r for r in g if r["measurable"]]
        if ok:
            def mean(key):
                vals = [r[key] for r in ok if r[key] is not None]
                return sum(vals) / len(vals) if vals else None
            out(f"    측정된 seed {len(ok)}/{len(g)} 평균: 주기 {_fmt(mean('period'))} step, "
                f"지연 {_fmt(mean('lag'))} step (주기의 {_fmt(mean('lag_ratio'), '.2f')}), "
                f"교차상관 {_fmt(mean('xcorr_max'), '.2f')}, "
                f"진폭 피식 {_fmt(mean('amplitude_prey'), '.1f')}배 / 포식 "
                f"{_fmt(mean('amplitude_predator'), '.1f')}배, 주기 수 {_fmt(mean('cycles'), '.1f')}, "
                f"감소 구간 포식 사망 비율 {_fmt(mean('predation_share'), '.2f')}")
        else:
            out(f"    측정된 seed 0/{len(g)}: 반복 진동 측정 불가")


def run_lv(policy="rule", seeds=None, steps=None, first_seed=1000, immigration=False,
           tables=None, results_dir=None, save=True, out=print):
    """policy 하나를 seed 여러 개로 돌려 LV 지표를 재고 CSV로 저장한다. 요약 행 목록을 돌려준다.

    저장: lv_<policy>.csv(seed별 시계열), lv_summary.csv(같은 policy의 예전 행만 바꿔 넣는다).
    immigration=False(기본)면 실행하는 동안 외부 유입을 끈다.
    """
    seeds = LV_SEEDS if seeds is None else seeds
    steps = LV_STEPS if steps is None else steps
    if policy == "trained" and tables is None:
        from .simulation import load_tables
        tables = load_tables()
    keys = series_keys()
    summary, series = [], []
    saved = config.IMMIGRATION_ENABLED
    config.IMMIGRATION_ENABLED = bool(immigration)
    try:
        for seed in range(first_seed, first_seed + seeds):
            rows = run_series(policy, seed, steps, tables)
            series.append((seed, rows))
            for (c, p), res in analyze_series(rows).items():
                summary.append(dict({"policy": policy, "seed": seed, "predator": c, "prey": p},
                                    **res))
    finally:
        config.IMMIGRATION_ENABLED = saved
    summary = [{k: r[k] for k in SUMMARY_COLUMNS} for r in summary]
    if out:
        out(f"\n[LV] 로트카-볼테라 유사성 — policy={policy}, seed {seeds}개 × {steps} step, "
            f"외부 유입 {'켬' if immigration else '끔'} "
            f"(과도 구간 {BURN_IN} step 제외, 이동 평균 {SMOOTH_WINDOW} step)")
        print_lv(summary, out)
    if save:
        results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"lv_{policy}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seed"] + keys)
            for seed, rows in series:
                for r in rows:
                    writer.writerow([seed] + [r[k] for k in keys])
        path = results_dir / "lv_summary.csv"
        old = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and list(reader.fieldnames) == list(SUMMARY_COLUMNS):
                    old = [r for r in reader if r["policy"] != policy]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(old)
            for r in summary:
                writer.writerow({k: ("" if v is None else round(v, 4) if isinstance(v, float) else v)
                                 for k, v in r.items()})
        if out:
            out(f"CSV 저장: {results_dir / f'lv_{policy}.csv'}, {path}")
    return summary
