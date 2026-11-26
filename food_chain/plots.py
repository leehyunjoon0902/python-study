"""평가·학습 결과 CSV로 발표용 차트(PNG)를 만든다. 종 목록은 CSV 헤더에서 찾는다.

사용법:
    uv run --no-sync python -c "from food_chain import plots; print(plots.make_all())"
    (또는 python -m food_chain --plots)

입력(없는 파일·열은 건너뜀):
    <results_dir>/eval_<policy>.csv   열: seed, turn, stats() 키 전체(count_<종>, grass_ratio, browse_ratio ...)
    <results_dir>/train_log.csv       열: episode, epsilon, mean_reward_<종>, ...
    <results_dir>/lv_<policy>.csv     열: seed, turn, count_<종>, deaths_*, kills_* (LV 측정, 유입 끔)
    <results_dir>/lv_summary.csv      정책 · seed · 쌍별 measurable, period, lag, lag_ratio, xcorr_max ...
    <results_dir>/noimm/eval_*.csv    외부 유입을 끈 평가(있으면 plots/noimm/ 에 같은 차트)
출력: <results_dir>/plots/*.png
    population_<policy>.png   초식(왼쪽 축)·육식(오른쪽 축) 개체수 + 아래 칸 풀·관목 비율
    oscillation_<policy>.png  종별 개체수 ÷ 자기 평균(평균 = 1) — 포식자의 시차 비교
    phase_<policy>.png        포식자–피식자 쌍별 위상도(여우–토끼, 늑대–사슴)
    needs_<policy>.png        종별 평균 배고픔·목마름(행 = 초식/육식)
    compare_policies.png      정책 비교(초식·육식 평균 개체수, 멸종 비율, 사망 원인)
    training_curve.png        초식/육식 보상 곡선 + ε
    lv_timeseries_<policy>.png  대표 seed의 여우–토끼 · 늑대–사슴 시계열(피식자 왼쪽 축, 포식자 오른쪽 축)
    lv_seeds_<policy>.png       모든 seed의 같은 시계열(소형 격자)
    lv_phase_<policy>.png       피식자 수 vs 포식자 수 궤적(회전 방향 표시)
    lv_summary.png              정책 · 쌍별 주기, 지연/주기(LV 이론 0.25 기준선), 교차상관 최대값
    noimm/*.png                 noimm/eval_*.csv 로 만든 같은 평가 차트(제목에 '외부 유입 끔')

원칙: 단위가 다른 값(개체 수 vs 풀 비율, 보상 vs ε)은 x축을 공유하는 위·아래 패널로
나눈다. 예외는 개체수 그래프의 초식/육식 두 축(계약서 v2 요구) — 축 제목에 어느 쪽 눈금인지 적는다.
"""

from __future__ import annotations

import csv
import os
import statistics
import traceback

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

try:   # 코어 설정(없거나 깨져 있어도 차트는 그린다)
    from food_chain import config as _config
except Exception:  # pragma: no cover
    _config = None

POLICIES = ('random', 'rule', 'trained')
POLICY_NAME = {'random': '무작위', 'rule': '규칙 기반', 'trained': '학습된 정책'}
KNOWN_ORDER = ('rabbit', 'deer', 'fox', 'wolf')
SPECIES_NAME = {'rabbit': '토끼', 'deer': '사슴', 'fox': '여우', 'wolf': '늑대',
                **dict(getattr(_config, 'SPECIES_LABEL', {}) or {})}
PREY = dict(getattr(_config, 'PREY', None) or {'fox': ('rabbit',), 'wolf': ('deer', 'rabbit')})
_HERB = tuple(getattr(_config, 'HERBIVORES', ()) or ('rabbit', 'deer'))
_CARN = tuple(getattr(_config, 'CARNIVORES', ()) or ('fox', 'wolf'))

# 색: 종 정체성은 모든 그림에서 같은 색(viewer.LINE_COLOR 와 같은 값).
# 초식 = 파랑·보라(차가운 계열), 육식 = 주황·적갈(따뜻한 계열)
COLOR = {'rabbit': '#2a78d6', 'deer': '#7048c8', 'fox': '#eb6834', 'wolf': '#a3242f',
         'grass': '#008300', 'browse': '#8a6a00', 'hunger': '#eda100', 'thirst': '#1baf7a'}
# 제목 뒤에 붙는 실험 조건 메모(예: noimm 폴더 차트 = ' · 외부 유입 끔'). make_all 이 바꾼다.
_NOTE = {'title': ''}


def _pname(policy):
    """정책 이름 + 실험 조건 메모(차트 제목용)."""
    return POLICY_NAME.get(policy, policy) + _NOTE['title']
_FALLBACK = ('#1baf7a', '#d6336c', '#495057', '#e8a400')
CAUSES = (('deaths_starvation', '아사', '#eda100'), ('deaths_dehydration', '탈수', '#1baf7a'),
          ('deaths_age', '노화', '#6250d6'), ('deaths_predation', '포식', '#e34948'))
INK = '#222222'
MUTED = '#6b6b6b'
DANGER = 35   # 뷰어와 같은 위험 기준(배고픔/목마름 65 이상)


def is_carnivore(sp):
    if sp in _CARN:
        return True
    if sp in _HERB:
        return False
    return sp in PREY


def _name(sp):
    return SPECIES_NAME.get(sp, sp)


def _color(sp):
    if sp not in COLOR:
        others = [s for s in COLOR if s not in KNOWN_ORDER]
        COLOR[sp] = _FALLBACK[len(others) % len(_FALLBACK)]
    return COLOR[sp]


def species_from_columns(columns, prefix='count_'):
    """헤더에서 '<prefix><종>' 열을 찾아 종 목록을 돌려준다(알려진 종 순서 우선)."""
    found = [c[len(prefix):] for c in columns if c.startswith(prefix) and len(c) > len(prefix)]
    return tuple(sorted(found, key=lambda s: (KNOWN_ORDER.index(s) if s in KNOWN_ORDER
                                              else len(KNOWN_ORDER), s)))


def _split(species):
    return (tuple(s for s in species if not is_carnivore(s)),
            tuple(s for s in species if is_carnivore(s)))


def _korean_font():
    """설치된 한글 글꼴(Windows: Malgun Gothic)을 고른다. 없으면 기본 글꼴."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR'):
        if name in names:
            return name
    return 'DejaVu Sans'


def _setup_style():
    plt.rcParams.update({
        'font.family': _korean_font(),
        'axes.unicode_minus': False,
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#9a9a9a',
        'axes.grid': True,
        'grid.color': '#e6e6e6',
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'figure.facecolor': 'white',
        'text.color': INK,
        'axes.labelcolor': INK,
    })


# ---------------------------------------------------------------- CSV 읽기
def _num(v):
    """CSV 문자열을 int → float 순으로 변환(안 되면 그대로)."""
    for conv in (int, float):
        try:
            return conv(v)
        except (TypeError, ValueError):
            pass
    return v


def _read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = [{k: _num(v) for k, v in r.items() if k is not None} for r in csv.DictReader(f)]
    return rows or None


def _by_seed(rows):
    runs = {}
    for r in rows:
        runs.setdefault(r.get('seed', 0), []).append(r)
    for v in runs.values():
        v.sort(key=lambda r: r.get('turn', 0))
    # seed 정렬: 숫자 seed 먼저(작은 순), 그 밖은 문자열 순
    return dict(sorted(runs.items(), key=lambda kv: (not isinstance(kv[0], (int, float)),
                                                     kv[0] if isinstance(kv[0], (int, float)) else 0,
                                                     str(kv[0]))))


def _rep_seed(runs):
    """대표 seed: 가장 작은 seed 번호(재현 가능하도록 고정)."""
    return next(iter(runs))


def _col(run, key, default=0.0):
    out = []
    for r in run:
        v = r.get(key, default)
        try:
            out.append(float(v if v not in ('', None) else 0.0))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _has(runs, key):
    return any(key in r for run in runs.values() for r in run[:1])


def _has_values(run, key):
    """한 seed의 행들에 key 열이 있고 빈 값이 아닌 행이 하나라도 있는가."""
    return any(r.get(key) not in (None, '') for r in run)


def _runs_species(runs):
    cols = set()
    for run in runs.values():
        if run:
            cols.update(run[0].keys())
    return species_from_columns(cols)


def _save(fig, out_dir, name, saved):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)


def _end_label(ax, x, y, text, color, dy=0):
    """선 끝에 이름을 직접 적는다(범례와 함께, 색에만 의존하지 않도록)."""
    if not x:
        return
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords='offset points',
                va='center', fontsize=12, color=color, fontweight='bold')
    ax.plot([x[-1]], [y[-1]], 'o', color=color, ms=5)


def _smooth_k(n):
    """긴 평가(1,000 step 초과)는 정수 개체수의 1-step 깜빡임이 진동을 가려서 이동평균을 쓴다."""
    return 1 if n <= 1000 else max(5, n // 100)


def _centered_avg(vals, k):
    """중심 이동평균(가장자리는 남은 점만 평균). 정점 위치가 뒤로 밀리지 않게 중심형을 쓴다."""
    if k <= 1 or not vals:
        return list(vals)
    pre = [0.0]
    for v in vals:
        pre.append(pre[-1] + v)
    h, n = k // 2, len(vals)
    return [(pre[min(n, i + h + 1)] - pre[max(0, i - h)]) / (min(n, i + h + 1) - max(0, i - h))
            for i in range(n)]


def _lw(n):
    """점이 많을수록 선을 가늘게(4,000 step 이상에서도 진동이 뭉개지지 않게)."""
    return 2.0 if n <= 600 else (1.4 if n <= 2500 else 1.1)


# ---------------------------------------------------------------- 개별 차트
def plot_population(policy, runs, out_dir, saved):
    """초식 = 왼쪽 축, 육식 = 오른쪽 축(수가 적은 육식의 오르내림이 보이게) + 풀 비율."""
    seed = _rep_seed(runs)
    run = runs[seed]
    t = _col(run, 'turn')
    herb, carn = _split(_runs_species(runs))
    has_grass = _has(runs, 'grass_ratio')
    if has_grass:
        fig, (ax, axg) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1.1], 'hspace': 0.1})
    else:
        fig, ax = plt.subplots(figsize=(14, 5.6))
        axg = None
    lw = _lw(len(t))
    k = _smooth_k(len(t))
    handles = []
    for sp in herb:
        y = _col(run, f'count_{sp}')
        if k > 1:
            ax.plot(t, y, color=_color(sp), lw=0.6, alpha=0.18)
        h, = ax.plot(t, _centered_avg(y, k), color=_color(sp), lw=lw,
                     label=f'{_name(sp)} (초식 · 왼쪽 축)')
        handles.append(h)
    ax.set_ylabel('초식 개체 수 (마리) ← 왼쪽 눈금', color=COLOR['rabbit'] if herb else INK)
    ax.set_ylim(bottom=0)
    if carn:
        axr = ax.twinx()
        axr.spines['right'].set_visible(True)
        axr.grid(False)
        for sp in carn:
            y = _col(run, f'count_{sp}')
            if k > 1:
                axr.plot(t, y, color=_color(sp), lw=0.6, alpha=0.18)
            h, = axr.plot(t, _centered_avg(y, k), color=_color(sp), lw=lw + 0.3, ls=(0, (5, 1.5)),
                          label=f'{_name(sp)} (육식 · 오른쪽 축, 점선)')
            handles.append(h)
        axr.set_ylabel('육식 개체 수 (마리) → 오른쪽 눈금', color=COLOR['fox'])
        axr.set_ylim(bottom=0, top=max([1.0] + [max(_col(run, f'count_{s}') or [0])
                                               for s in carn]) * 1.12)
        axr.tick_params(axis='y', colors='#8a3a12')
    ax.set_ylim(top=max([1.0] + [max(_col(run, f'count_{s}') or [0]) for s in herb]) * 1.12)
    ax.tick_params(axis='y', colors='#1d4f8f')
    smooth = f' · 진한 선 = {k} step 중심 이동평균, 흐린 선 = 원값' if k > 1 else ''
    ax.set_title(f'개체수 변화 — {_pname(policy)} (seed {seed}, '
                 f'{int(t[-1]) if t else 0} step){smooth}', loc='left')
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.06), frameon=False,
              ncol=min(4, len(handles)), fontsize=11, handlelength=2.6)
    if t:
        ax.set_xlim(t[0], t[-1])

    if axg is not None:
        g = [v * 100 for v in _col(run, 'grass_ratio')]
        axg.plot(t, g, color=COLOR['grass'], lw=lw, label='풀 (토끼 먹이)')
        axg.fill_between(t, g, color=COLOR['grass'], alpha=0.10, lw=0)
        has_browse = _has_values(run, 'browse_ratio')
        if has_browse:   # 관목(사슴 먹이층): 같은 단위(%)라 같은 칸에 점선으로
            b = [v * 100 for v in _col(run, 'browse_ratio')]
            axg.plot(t, b, color=COLOR['browse'], lw=lw + 0.2, ls=(0, (5, 1.5)),
                     label='관목 (사슴 먹이, 점선)')
        axg.set_ylim(0, 100)
        axg.set_ylabel('자란 정도 (%)' if has_browse else '풀 비율 (%)')
        axg.set_xlabel('turn (step)')
        if has_browse:
            axg.legend(loc='lower right', frameon=False, ncol=2, fontsize=11)
    else:
        ax.set_xlabel('turn (step)')
    _save(fig, out_dir, f'population_{policy}.png', saved)


def _best_lag(prey, pred, max_lag):
    """포식자 곡선이 피식자 곡선을 몇 step 늦게 따라가는지(상호상관 최대 시차, ≥ 0)."""
    n = min(len(prey), len(pred))
    if n < 20:
        return None
    a, b = prey[:n], pred[:n]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    a = [v - ma for v in a]
    b = [v - mb for v in b]
    sa = sum(v * v for v in a) ** 0.5
    sb = sum(v * v for v in b) ** 0.5
    if sa == 0 or sb == 0:
        return None
    step = max(1, max_lag // 200)
    best, best_r = None, -2.0
    for lag in range(0, max_lag + 1, step):
        r = sum(a[i] * b[i + lag] for i in range(n - lag)) / (sa * sb)
        if r > best_r:
            best, best_r = lag, r
    return best, best_r


def plot_oscillation(policy, runs, out_dir, saved):
    """각 종의 개체수 ÷ 자기 평균(평균 = 1). 단위가 같아져 한 축에서 시차를 비교할 수 있다."""
    seed = _rep_seed(runs)
    run = runs[seed]
    t = _col(run, 'turn')
    species = _runs_species(runs)
    fig, ax = plt.subplots(figsize=(14, 5.8))
    lw = _lw(len(t)) + 0.3
    k = _smooth_k(len(t))
    ymax = 2.0
    for sp in species:
        y = _col(run, f'count_{sp}')
        m = statistics.fmean(y) if y else 0.0
        if m <= 0:
            continue
        rel = _centered_avg([v / m for v in y], k)
        ymax = max(ymax, max(rel))
        carn = is_carnivore(sp)
        ax.plot(t, rel, color=_color(sp), lw=lw + (0.3 if carn else 0),
                ls=(0, (5, 1.5)) if carn else '-',
                label=f'{_name(sp)} ({"육식" if carn else "초식"}, 평균 {m:.1f}마리)')
    ax.axhline(1.0, color='#7a7a7a', lw=1.0, ls=':')
    ax.annotate('평균 = 1', (1.0, 1.0), xycoords=('axes fraction', 'data'), xytext=(4, 0),
                textcoords='offset points', va='center', fontsize=11, color=MUTED)
    ax.set_ylim(0, min(ymax * 1.05, 6.0))
    if t:
        ax.set_xlim(t[0], t[-1])
    ax.set_ylabel('개체 수 ÷ 그 종의 평균')
    ax.set_xlabel('turn (step)')
    # 포식자가 피식자를 몇 step 늦게 따라가는지 상호상관으로 추정해 적는다
    notes = []
    for pred, prey in _pairs(species):
        res = _best_lag(_centered_avg(_col(run, f'count_{prey}'), k),
                        _centered_avg(_col(run, f'count_{pred}'), k),
                        max(10, len(t) // 4))
        if res:
            notes.append(f'{_name(pred)}는 {_name(prey)}보다 약 {res[0]} step 늦게 따라감 '
                         f'(상관 {res[1]:.2f})')
    if notes:
        ax.text(0.995, 0.97, '\n'.join(notes), transform=ax.transAxes, ha='right', va='top',
                fontsize=11, color=INK,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', alpha=0.9))
    smooth = f' · {k} step 이동평균' if k > 1 else ''
    ax.set_title(f'상대 개체수 진동 — {_pname(policy)} (seed {seed}) · '
                 f'실선 = 초식, 점선 = 육식{smooth}', loc='left')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.06), frameon=False,
              ncol=min(4, len(species)), fontsize=11, handlelength=2.6)
    _save(fig, out_dir, f'oscillation_{policy}.png', saved)


def _pairs(species):
    """(포식자, 대표 먹이) 쌍: config.PREY 에서 각 포식자의 첫 번째 먹이."""
    out = []
    for pred in species:
        if not is_carnivore(pred):
            continue
        prey = [p for p in PREY.get(pred, ()) if p in species]
        if not prey:   # PREY 에 없으면 아무 초식이나
            prey = [s for s in species if not is_carnivore(s)][:1]
        if prey:
            out.append((pred, prey[0]))
    return out


def plot_phase(policy, runs, out_dir, saved):
    pairs = _pairs(_runs_species(runs))
    if not pairs:
        return
    seed = _rep_seed(runs)
    fig, axes = plt.subplots(1, len(pairs), figsize=(8.2 * len(pairs) + 1, 7), squeeze=False)
    axes = axes[0]
    cmap = ListedColormap(plt.get_cmap('Blues')([0.3 + 0.7 * i / 255 for i in range(256)]))
    run = runs[seed]
    t = _col(run, 'turn')
    k = _smooth_k(len(t))   # 긴 평가는 정수 개체수의 계단 잡음을 이동평균으로 눌러 고리가 보이게
    lc = None
    for ax, (pred, prey) in zip(axes, pairs):
        # 다른 seed는 흐린 회색으로 배경에
        for s, other in runs.items():
            if s == seed:
                continue
            ax.plot(_centered_avg(_col(other, f'count_{prey}'), k),
                    _centered_avg(_col(other, f'count_{pred}'), k), color='#c8c8c8',
                    lw=0.8, alpha=0.7, zorder=1)
        x = _centered_avg(_col(run, f'count_{prey}'), k)
        y = _centered_avg(_col(run, f'count_{pred}'), k)
        if len(x) >= 2:
            pts = list(zip(x, y))
            segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
            lc = LineCollection(segs, cmap=cmap, lw=1.6 if len(segs) > 1500 else 2, zorder=2)
            lc.set_array([t[i] for i in range(len(segs))])
            lc.set_clim(min(t), max(t) if max(t) > min(t) else min(t) + 1)
            ax.add_collection(lc)
        if x:
            ax.plot(x[0], y[0], 'o', ms=10, color='white', mec=INK, mew=2, zorder=3, label='시작')
            ax.plot(x[-1], y[-1], 'X', ms=11, color=INK, zorder=3, label='끝')
        ax.set_xlim(left=0, right=max([1] + [max(_col(r, f'count_{prey}') or [0])
                                             for r in runs.values()]) * 1.05)
        ax.set_ylim(bottom=0, top=max([1] + [max(_col(r, f'count_{pred}') or [0])
                                             for r in runs.values()]) * 1.08)
        ax.set_xlabel(f'{_name(prey)} 수 (마리, 먹이)', color=_color(prey))
        ax.set_ylabel(f'{_name(pred)} 수 (마리, 포식자)', color=_color(pred))
        ax.set_title(f'{_name(pred)} – {_name(prey)}', loc='left')
        if len(runs) > 1:
            ax.plot([], [], color='#c8c8c8', lw=1.5, label='다른 seed')
        ax.legend(loc='upper right', frameon=False)
    if lc is not None:
        cb = fig.colorbar(lc, ax=list(axes), pad=0.02, fraction=0.03 if len(pairs) > 1 else 0.05)
        cb.set_label('turn (진할수록 나중)')
        cb.outline.set_visible(False)
    fig.suptitle(f'포식자–피식자 위상도 — {_pname(policy)} (seed {seed}) · '
                 f'반시계 방향 고리 = 포식자가 먹이를 시차를 두고 따라감'
                 + (f' · 두 축 모두 {k} step 이동평균' if k > 1 else ''),
                 x=0.06, ha='left', fontweight='bold', fontsize=15)
    _save(fig, out_dir, f'phase_{policy}.png', saved)


def plot_needs(policy, runs, out_dir, saved):
    """seed 평균(해당 종이 살아 있는 seed만) 배고픔·목마름. 행 = 초식/육식."""
    herb, carn = _split(_runs_species(runs))
    rows = [g for g in (herb, carn) if g]
    if not rows:
        return
    ncol = max(len(g) for g in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(6.6 * ncol + 0.5, 4.4 * len(rows)),
                             sharey=True, squeeze=False)
    for i, group in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(group):
                ax.set_visible(False)
                continue
            sp = group[j]
            acc = {}
            for run in runs.values():
                for r in run:
                    if float(r.get(f'count_{sp}', 0) or 0) <= 0:
                        continue
                    d = acc.setdefault(r.get('turn', 0), [[], []])
                    d[0].append(100 - float(r.get(f'avg_energy_{sp}', 0) or 0))
                    d[1].append(100 - float(r.get(f'avg_hydration_{sp}', 0) or 0))
            turns = sorted(acc)
            hunger = [statistics.fmean(acc[k][0]) for k in turns]
            thirst = [statistics.fmean(acc[k][1]) for k in turns]
            ax.axhspan(100 - DANGER, 100, color='#e34948', alpha=0.07, lw=0)
            ax.text(0.01, (100 - DANGER + 100) / 2 / 100, '위험 구간', transform=ax.transAxes,
                    color='#b02525', fontsize=11, va='center')
            lw = _lw(len(turns))
            ax.plot(turns, hunger, color=COLOR['hunger'], lw=lw, label='배고픔 (100 - energy)')
            ax.plot(turns, thirst, color=COLOR['thirst'], lw=lw, ls='--',
                    label='목마름 (100 - hydration)')
            dy = 0
            if turns and abs(hunger[-1] - thirst[-1]) < 7:   # 끝 이름표가 겹치면 위아래로 벌림
                dy = 7 if hunger[-1] >= thirst[-1] else -7
            _end_label(ax, turns, hunger, '배고픔', COLOR['hunger'], dy=dy)
            _end_label(ax, turns, thirst, '목마름', COLOR['thirst'], dy=-dy)
            kind = '육식' if is_carnivore(sp) else '초식'
            ax.set_title(f'{_name(sp)} ({kind})', loc='left', color=_color(sp))
            ax.set_ylim(0, 100)
            if i == len(rows) - 1 or j >= len(rows[-1]):
                ax.set_xlabel('turn (step)')
            if j == 0:
                ax.set_ylabel('평균 수치 (0~100)')
            if not turns:
                ax.text(0.5, 0.5, '데이터 없음(멸종)', transform=ax.transAxes, ha='center',
                        color=MUTED)
    axes[0][0].legend(loc='upper right', frameon=False, fontsize=11)
    fig.suptitle(f'종별 평균 배고픔·목마름 — {_pname(policy)} '
                 f'(seed {len(runs)}개 평균)', x=0.06, ha='left', fontweight='bold', fontsize=15)
    fig.tight_layout()
    _save(fig, out_dir, f'needs_{policy}.png', saved)


def _summary(runs, species):
    """정책 하나의 요약: 종별 평균 개체수, 멸종 seed 비율, 사망 원인 합계."""
    n = len(runs)
    out = {}
    for sp in species:
        means = [statistics.fmean(_col(run, f'count_{sp}')) for run in runs.values() if run]
        out[f'mean_{sp}'] = statistics.fmean(means) if means else 0.0
        out[f'sd_{sp}'] = statistics.pstdev(means) if len(means) > 1 else 0.0
        ext = sum(1 for run in runs.values() if run and float(run[-1].get(f'count_{sp}', 0) or 0) <= 0)
        out[f'ext_{sp}'] = ext / n if n else 0.0
    for key, _, _ in CAUSES:
        out[key] = sum(float(run[-1].get(key, 0) or 0) for run in runs.values() if run)
    return out


def _grouped_bars(ax, xs, pols, summ, species, key, fmt, scale=1.0, err=None):
    n = max(1, len(species))
    w = 0.8 / n
    for i, sp in enumerate(species):
        pos = [x + (i - (n - 1) / 2) * w for x in xs]
        vals = [summ[p][f'{key}_{sp}'] * scale for p in pols]
        kw = {}
        if err:
            kw = dict(yerr=[summ[p][f'{err}_{sp}'] for p in pols], ecolor='#555555', capsize=3,
                      error_kw={'lw': 1})
        bars = ax.bar(pos, vals, w * 0.92, color=_color(sp), label=_name(sp), **kw)
        ax.bar_label(bars, labels=[fmt(v) for v in vals], padding=2, fontsize=10)


def _immigrant_means(summary_rows, pols):
    """eval_summary.csv 행에서 정책별 · 종별 seed 평균 외부 유입 수.

    immigrants_<종> 열이 없으면 None. 열이 있어도 모든 정책에서 0인 종은 빼고,
    전부 0이면 열이 있는 종을 모두 보여 준다(유입 0 자체가 정보).
    """
    if not summary_rows:
        return None
    sp_all = species_from_columns(summary_rows[0].keys(), prefix='immigrants_')
    if not sp_all:
        return None
    means = {}
    for sp in sp_all:
        for p in pols:
            vals = []
            for r in summary_rows:
                if str(r.get('policy', '')) != p:
                    continue
                try:
                    vals.append(float(r.get(f'immigrants_{sp}') or 0))
                except (TypeError, ValueError):
                    pass
            if vals:
                means.setdefault(sp, {})[p] = statistics.fmean(vals)
    if not means:
        return None
    nonzero = {sp: d for sp, d in means.items() if any(v > 0 for v in d.values())}
    return nonzero or means


def plot_compare(evals, out_dir, saved, summary_rows=None):
    pols = [p for p in POLICIES if p in evals] + [p for p in evals if p not in POLICIES]
    species = species_from_columns({k for p in pols for run in evals[p].values()
                                    for k in (run[0].keys() if run else ())})
    herb, carn = _split(species)
    summ = {p: _summary(evals[p], species) for p in pols}
    names = [POLICY_NAME.get(p, p) for p in pols]
    xs = list(range(len(pols)))
    immig = _immigrant_means(summary_rows, pols)
    if immig:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.55], wspace=0.25, hspace=0.5)
        axes = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
        ax_imm = fig.add_subplot(gs[2, :])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                                 gridspec_kw={'wspace': 0.25, 'hspace': 0.42})
        ax_imm = None

    # (1)(2) 평균 개체수: 초식·육식은 규모가 달라 패널을 나눈다
    for ax, group, title in ((axes[0][0], herb, '초식 평균 개체수'),
                             (axes[0][1], carn, '육식 평균 개체수')):
        if not group:
            ax.set_visible(False)
            continue
        _grouped_bars(ax, xs, pols, summ, group, 'mean', lambda v: f'{v:.1f}', err='sd')
        ax.set_xticks(xs, names)
        ax.set_ylabel('시간 평균 개체 수 (마리)')
        ax.set_title(title, loc='left')
        ax.margins(y=0.18)
        ax.grid(axis='x', visible=False)
        ax.legend(frameon=False, loc='upper left', ncol=len(group))

    # (3) 멸종한 seed 비율
    ax = axes[1][0]
    _grouped_bars(ax, xs, pols, summ, species, 'ext', lambda v: f'{v:.0f}%', scale=100)
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 118)
    ax.set_ylabel('멸종한 seed 비율 (%)')
    ax.set_title('멸종 비율 (마지막 turn 기준)', loc='left')
    ax.legend(frameon=False, loc='upper left', ncol=len(species), fontsize=11)
    ax.grid(axis='x', visible=False)

    # (4) 사망 원인 구성(100% 누적 막대)
    ax = axes[1][1]
    bottoms = [0.0] * len(pols)
    for key, label, color in CAUSES:
        shares = []
        for p in pols:
            tot = sum(summ[p][k] for k, _, _ in CAUSES)
            shares.append(summ[p][key] / tot * 100 if tot else 0.0)
        bars = ax.bar(xs, shares, 0.6, bottom=bottoms, color=color, label=label,
                      edgecolor='white', linewidth=2)
        for b, s, bt in zip(bars, shares, bottoms):
            if s >= 7:
                ax.text(b.get_x() + b.get_width() / 2, bt + s / 2, f'{s:.0f}%', ha='center',
                        va='center', fontsize=11,
                        color='white' if key in ('deaths_age', 'deaths_predation') else INK,
                        fontweight='bold')
        bottoms = [b + s for b, s in zip(bottoms, shares)]
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 100)
    ax.set_ylabel('사망 원인 비율 (%)')
    ax.set_title('사망 원인 구성 (모든 seed · 모든 종 합계)', loc='left')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis='x', visible=False)

    # (5) 외부 유입(eval_summary.csv 의 immigrants_<종>, 있을 때만): 가로 막대로 작게
    if ax_imm is not None:
        isp = [s for s in KNOWN_ORDER if s in immig] + sorted(s for s in immig if s not in KNOWN_ORDER)
        n = len(isp)
        h = 0.8 / max(1, n)
        for i, sp in enumerate(isp):
            pos = [x + (i - (n - 1) / 2) * h for x in xs]
            vals = [immig[sp].get(p, 0.0) for p in pols]
            bars = ax_imm.barh(pos, vals, h * 0.9, color=_color(sp), label=_name(sp))
            ax_imm.bar_label(bars, labels=[f'{v:.1f}' for v in vals], padding=3, fontsize=10)
        ax_imm.set_yticks(xs, names)
        ax_imm.invert_yaxis()
        ax_imm.margins(x=0.08)
        ax_imm.set_xlabel('seed당 평균 유입 수 (마리, 누적)')
        ax_imm.set_title('외부 유입 — 육식 수가 기준 아래로 떨어지면 가장자리에서 들어온 개체 수 '
                         '(eval_summary.csv)', loc='left')
        ax_imm.legend(frameon=False, loc='lower right', ncol=n)
        ax_imm.grid(axis='y', visible=False)

    nseeds = ', '.join(f'{POLICY_NAME.get(p, p)} {len(evals[p])}' for p in pols)
    fig.suptitle(f'정책 비교{_NOTE["title"]} (seed 수: {nseeds})', x=0.06, ha='left', fontweight='bold',
                 fontsize=16, y=0.98)
    _save(fig, out_dir, 'compare_policies.png', saved)


def _moving_avg(vals, k):
    out, s = [], 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= k:
            s -= vals[i - k]
        out.append(s / min(i + 1, k))
    return out


def plot_training(rows, out_dir, saved):
    ep = [float(r.get('episode', i)) for i, r in enumerate(rows)]
    k = max(1, len(rows) // 20)
    species = species_from_columns(rows[0].keys(), prefix='mean_reward_')
    herb, carn = _split(species)
    groups = [(g, t) for g, t in ((herb, '초식'), (carn, '육식')) if g]
    has_eps = 'epsilon' in rows[0]
    n_panels = len(groups) + (1 if has_eps else 0)
    if n_panels == 0:
        return
    ratios = [3] * len(groups) + ([1.2] if has_eps else [])
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.2 * len(groups) + (1.8 if has_eps else 0)),
                             sharex=True, squeeze=False,
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.18})
    axes = [a[0] for a in axes]
    for ax, (group, title) in zip(axes, groups):
        ends = []
        for sp in group:
            y = [float(r.get(f'mean_reward_{sp}', 0) or 0) for r in rows]
            ma = _moving_avg(y, k)
            ax.plot(ep, y, color=_color(sp), lw=0.8, alpha=0.25)
            ax.plot(ep, ma, color=_color(sp), lw=2.2, label=f'{_name(sp)} (이동평균 {k})')
            ends.append((ma, sp, y))
        # 끝 이름표: 값이 가까우면(축 범위의 8% 이내) 위아래로 벌려 겹치지 않게
        span = max((max(v[2]) - min(v[2]) for v in ends if v[2]), default=1) or 1
        ends.sort(key=lambda v: v[0][-1] if v[0] else 0)
        for i, (ma, sp, _) in enumerate(ends):
            close = any(abs(ma[-1] - o[0][-1]) < span * 0.08 for o in ends if o[1] != sp) \
                if ma else False
            dy = (i - (len(ends) - 1) / 2) * 13 if close else 0
            _end_label(ax, ep, ma, _name(sp), _color(sp), dy=dy)
        ax.set_ylabel(f'{title} episode\n평균 보상')
        ax.axhline(0, color='#9a9a9a', lw=0.8)
        ax.legend(frameon=False, loc='upper left', ncol=len(group), fontsize=11)
    if groups:
        axes[0].set_title('학습 곡선 — 종별 평균 보상 (초식/육식 칸 분리)', loc='left')
    if has_eps:
        axe = axes[-1]
        eps = [float(r.get('epsilon', 0) or 0) for r in rows]
        axe.plot(ep, eps, color='#52514e', lw=2)
        axe.set_ylabel('ε (탐험률)')
        axe.set_ylim(0, max(eps + [0.01]) * 1.1)
    axes[-1].set_xlabel('episode')
    _save(fig, out_dir, 'training_curve.png', saved)


# ---------------------------------------------------------------- 로트카-볼테라(LV) 차트
LV_POLICIES = ('rule', 'trained')
LV_SMOOTH = 101          # analysis.SMOOTH_WINDOW 와 같은 평활 창(step)
LV_BURN_IN = 1000        # analysis.BURN_IN: 분석에서 뺀 초기 과도 구간
LV_THEORY = 0.25         # LV 이론의 포식자 지연 / 주기
LV_REP_SEED = 1000       # 대표 seed(있으면)
try:   # 분석 모듈의 값과 맞춘다(없어도 위 기본값으로 그린다)
    from food_chain import analysis as _analysis
    LV_SMOOTH = int(getattr(_analysis, 'SMOOTH_WINDOW', LV_SMOOTH))
    LV_BURN_IN = int(getattr(_analysis, 'BURN_IN', LV_BURN_IN))
    _LV_PAIRS = tuple(getattr(_analysis, 'PAIRS', ()))
except Exception:  # pragma: no cover
    _analysis = None
    _LV_PAIRS = ()


def _lv_pairs(species):
    """LV 분석 쌍 (포식자, 피식자): analysis.PAIRS → 없으면 config.PREY 의 첫 먹이."""
    pairs = [(c, p) for c, p in (_LV_PAIRS or ()) if c in species and p in species]
    return pairs or _pairs(species)


def _lv_summary_index(rows):
    """lv_summary.csv 행 → {(policy, seed, predator): 행}."""
    out = {}
    for r in rows or ():
        out[str(r.get('policy', '')), r.get('seed'), str(r.get('predator', ''))] = r
    return out


def _fnum(v):
    """빈 값이면 None, 아니면 float."""
    if v in (None, ''):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _lv_note(srow, short=False):
    """요약 행 하나를 사람이 읽는 한 줄로."""
    if not srow:
        return ''
    ext = _fnum(srow.get('extinct_step'))
    ext_txt = f'{int(ext)} step에 한쪽 멸종' if ext is not None and ext >= 0 else ''
    period, lag, ratio = (_fnum(srow.get(k)) for k in ('period', 'lag', 'lag_ratio'))
    xc = _fnum(srow.get('xcorr_max'))
    if int(_fnum(srow.get('measurable')) or 0) == 1 and period:
        if short:
            txt = f'주기 {period:.0f} · 지연 {lag:.0f} (지연/주기 {ratio:.2f})'
        else:
            txt = (f'주기 {period:.0f} step · 포식자 지연 {lag:.0f} step '
                   f'(지연/주기 {ratio:.2f}, LV 이론 {LV_THEORY:g}) · 교차상관 {xc:.2f}')
    else:
        txt = '주기 측정 불가(뚜렷한 반복 2회 미만)'
        if lag is not None and not short:
            txt += f' · 교차상관 최대 지연 {lag:.0f} step (r = {xc:.2f})'
    return f'{txt} · {ext_txt}' if ext_txt else txt


def _lv_rep_seed(runs, pairs):
    """대표 seed: LV_REP_SEED → 두 쌍 모두 끝까지 살아남은 첫 seed → 첫 seed."""
    if LV_REP_SEED in runs:
        return LV_REP_SEED
    for s, run in runs.items():
        if all(min(_col(run, f'count_{sp}')) > 0 for pair in pairs for sp in pair):
            return s
    return next(iter(runs))


def _peaks(x, half_width):
    """평활한 시계열의 봉우리 위치(analysis.count_peaks 가 있으면 그것을 쓴다)."""
    if _analysis is not None and hasattr(_analysis, 'count_peaks'):
        try:
            return list(_analysis.count_peaks(x, half_width))
        except Exception:  # pragma: no cover
            pass
    out = []
    n = len(x)
    for i in range(half_width, n - half_width):
        win = x[i - half_width:i + half_width + 1]
        if x[i] == max(win) and x[i] > min(win):
            if not out or i - out[-1] > half_width:
                out.append(i)
    return out


def _lv_pair_axes(ax, t, prey_raw, pred_raw, prey, pred, small=False, peaks_hw=None):
    """한 쌍을 그린다: 피식자 = 왼쪽 축(실선), 포식자 = 오른쪽 축(점선). 오른쪽 축을 돌려준다."""
    lw = 1.1 if small else 1.8
    if not small:
        ax.plot(t, prey_raw, color=_color(prey), lw=0.5, alpha=0.15)
    ax.plot(t, _centered_avg(prey_raw, LV_SMOOTH), color=_color(prey), lw=lw)
    axr = ax.twinx()
    axr.spines['right'].set_visible(True)
    axr.grid(False)
    ps = _centered_avg(pred_raw, LV_SMOOTH)
    axr.plot(t, ps, color=_color(pred), lw=lw + 0.3, ls=(0, (5, 1.5)))
    ax.set_ylim(0, max([1.0] + (prey_raw if not small else _centered_avg(prey_raw, LV_SMOOTH)))
                * 1.08)
    axr.set_ylim(0, max([1.0] + ps) * 1.15)
    ax.tick_params(axis='y', colors=_color(prey), labelsize=9 if small else 12)
    axr.tick_params(axis='y', colors=_color(pred), labelsize=9 if small else 12)
    if peaks_hw:   # 봉우리 표시: 피식자 ▲, 포식자 ▼ — 포식자 봉우리가 뒤에 오는지 보이게
        ys = _centered_avg(prey_raw, LV_SMOOTH)
        start = next((i for i, v in enumerate(t) if v >= LV_BURN_IN), 0)
        pk = [i + start for i in _peaks(ys[start:], peaks_hw)]
        ax.plot([t[i] for i in pk], [ys[i] for i in pk], '^', color=_color(prey), ms=9,
                mec='white', mew=1.2, zorder=4)
        qk = [i + start for i in _peaks(ps[start:], peaks_hw)]
        axr.plot([t[i] for i in qk], [ps[i] for i in qk], 'v', color=_color(pred), ms=9,
                 mec='white', mew=1.2, zorder=4)
    return axr


def plot_lv_timeseries(policy, runs, summ, out_dir, saved):
    """대표 seed 한 개의 여우–토끼 · 늑대–사슴 시계열(위아래 칸)."""
    species = _runs_species(runs)
    pairs = _lv_pairs(species)
    if not pairs:
        return
    seed = _lv_rep_seed(runs, pairs)
    run = runs[seed]
    t = _col(run, 'turn')
    fig, axes = plt.subplots(len(pairs), 1, figsize=(16, 4.6 * len(pairs) + 1.2), sharex=True,
                             squeeze=False, gridspec_kw={'hspace': 0.42})
    for ax, (pred, prey) in zip((a[0] for a in axes), pairs):
        srow = summ.get((policy, seed, pred))
        period = _fnum(srow.get('period')) if srow else None
        hw = max(1, int(period * 0.35)) if period else None
        axr = _lv_pair_axes(ax, t, _col(run, f'count_{prey}'), _col(run, f'count_{pred}'),
                            prey, pred, peaks_hw=hw)
        ax.axvspan(t[0] if t else 0, LV_BURN_IN, color='#9a9a9a', alpha=0.12, lw=0)
        ax.text((t[0] if t else 0) + LV_BURN_IN * 0.06, 0.03, '초기\n과도 구간\n(분석 제외)',
                transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=9,
                color=MUTED)
        ax.set_ylabel(f'{_name(prey)} 수 (마리) · 왼쪽 눈금, 실선', color=_color(prey))
        axr.set_ylabel(f'{_name(pred)} 수 (마리) · 오른쪽 눈금, 점선', color=_color(pred))
        ax.set_title(f'{_name(pred)}–{_name(prey)}  ·  {_lv_note(srow)}', loc='left', fontsize=13)
        handles = [plt.Line2D([], [], color=_color(prey), lw=2,
                              label=f'{_name(prey)} (피식자, {LV_SMOOTH} step 이동평균)'),
                   plt.Line2D([], [], color=_color(pred), lw=2.3, ls=(0, (5, 1.5)),
                              label=f'{_name(pred)} (포식자, {LV_SMOOTH} step 이동평균)')]
        if hw:
            handles += [plt.Line2D([], [], color=_color(prey), marker='^', ls='', ms=9,
                                   label=f'{_name(prey)} 봉우리'),
                        plt.Line2D([], [], color=_color(pred), marker='v', ls='', ms=9,
                                   label=f'{_name(pred)} 봉우리')]
        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), frameon=False,
                  ncol=len(handles), fontsize=11, borderaxespad=1.6)
        if t:
            ax.set_xlim(t[0], t[-1])
    axes[-1][0].set_xlabel('turn (step)')
    fig.suptitle(f'로트카-볼테라 시계열 — {_pname(policy)} (seed {seed}, {int(t[-1]) if t else 0} step, '
                 f'외부 유입 끔) · 흐린 선 = 원값', x=0.06, ha='left', fontweight='bold',
                 fontsize=16, y=0.995)
    _save(fig, out_dir, f'lv_timeseries_{policy}.png', saved)


def plot_lv_seeds(policy, runs, summ, out_dir, saved):
    """모든 seed의 같은 시계열 소형 격자(행 = seed, 열 = 쌍)."""
    species = _runs_species(runs)
    pairs = _lv_pairs(species)
    if not pairs:
        return
    seeds = list(runs)
    fig, axes = plt.subplots(len(seeds), len(pairs),
                             figsize=(9 * len(pairs), 2.3 * len(seeds) + 1.6),
                             sharex=True, squeeze=False,
                             gridspec_kw={'hspace': 0.55, 'wspace': 0.28})
    for i, s in enumerate(seeds):
        run = runs[s]
        t = _col(run, 'turn')
        for j, (pred, prey) in enumerate(pairs):
            ax = axes[i][j]
            _lv_pair_axes(ax, t, _col(run, f'count_{prey}'), _col(run, f'count_{pred}'),
                          prey, pred, small=True)
            ax.axvspan(t[0] if t else 0, LV_BURN_IN, color='#9a9a9a', alpha=0.12, lw=0)
            ax.set_title(f'seed {s} · {_name(pred)}–{_name(prey)} · '
                         f'{_lv_note(summ.get((policy, s, pred)), short=True)}',
                         loc='left', fontsize=11, fontweight='normal')
            if t:
                ax.set_xlim(t[0], t[-1])
            ax.tick_params(axis='x', labelsize=9)
            if i == len(seeds) - 1:
                ax.set_xlabel('turn (step)', fontsize=11)
    handles = []
    for pred, prey in pairs:
        handles += [plt.Line2D([], [], color=_color(prey), lw=2,
                               label=f'{_name(prey)} (왼쪽 눈금, 실선)'),
                    plt.Line2D([], [], color=_color(pred), lw=2.3, ls=(0, (5, 1.5)),
                               label=f'{_name(pred)} (오른쪽 눈금, 점선)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.06, 0.985 - 0.35 / fig.get_figheight()),
               ncol=len(handles), frameon=False, fontsize=12)
    fig.suptitle(f'로트카-볼테라 시계열 — {_pname(policy)} · seed {len(seeds)}개 · '
                 f'{LV_SMOOTH} step 이동평균 · 회색 = 초기 과도 구간', x=0.06, ha='left',
                 fontweight='bold', fontsize=16, y=0.995)
    _save(fig, out_dir, f'lv_seeds_{policy}.png', saved)


def _signed_area(xs, ys):
    """닫히지 않은 궤적의 부호 면적(신발끈 공식). > 0 이면 반시계 방향(x = 피식자, y = 포식자)."""
    a = 0.0
    for i in range(len(xs) - 1):
        a += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return a / 2


def plot_lv_phase(policy, runs, summ, out_dir, saved):
    """피식자 수(x) vs 포식자 수(y) 궤적. 반시계 고리 = 포식자가 피식자를 뒤따른다."""
    species = _runs_species(runs)
    pairs = _lv_pairs(species)
    if not pairs:
        return
    seed = _lv_rep_seed(runs, pairs)
    fig, axes = plt.subplots(1, len(pairs), figsize=(8.4 * len(pairs) + 1, 7.6), squeeze=False)
    axes = axes[0]
    cmap = ListedColormap(plt.get_cmap('Blues')([0.3 + 0.7 * i / 255 for i in range(256)]))
    every = 5     # 그림용 솎아내기(평활 창 101보다 훨씬 작아 모양은 그대로)
    lc = None
    for ax, (pred, prey) in zip(axes, pairs):
        ccw = 0
        n_alive = 0
        rep_rot = '-'
        xmax = ymax = 1.0
        for s, run in runs.items():
            t = _col(run, 'turn')
            start = next((i for i, v in enumerate(t) if v >= LV_BURN_IN), 0)
            xr, yr = _col(run, f'count_{prey}'), _col(run, f'count_{pred}')
            end = next((i for i in range(len(xr)) if xr[i] <= 0 or yr[i] <= 0), len(xr))
            x = _centered_avg(xr[:end], LV_SMOOTH)[start::every]
            y = _centered_avg(yr[:end], LV_SMOOTH)[start::every]
            if len(x) < 3:
                continue
            n_alive += 1
            if _signed_area(x, y) > 0:
                ccw += 1
            xmax, ymax = max(xmax, max(x)), max(ymax, max(y))
            if s != seed:
                ax.plot(x, y, color='#c8c8c8', lw=0.8, alpha=0.8, zorder=1)
                continue
            tt = t[start:end:every][:len(x)]
            pts = list(zip(x, y))
            segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
            lc = LineCollection(segs, cmap=cmap, lw=1.8, zorder=2)
            lc.set_array(tt[:len(segs)])
            lc.set_clim(min(tt), max(tt) if max(tt) > min(tt) else min(tt) + 1)
            ax.add_collection(lc)
            # 진행 방향 화살표(궤적을 따라 일정 간격)
            n_arrow = 14
            for k in range(1, n_arrow + 1):
                i = int(k * (len(pts) - 2) / (n_arrow + 1))
                ax.annotate('', xy=pts[i + 1], xytext=pts[i], zorder=3,
                            arrowprops=dict(arrowstyle='-|>', color='#08306b', lw=1.2,
                                            mutation_scale=16))
            ax.plot(*pts[0], 'o', ms=10, color='white', mec=INK, mew=2, zorder=4,
                    label=f'시작 (turn {int(tt[0])})')
            ax.plot(*pts[-1], 'X', ms=11, color=INK, zorder=4, label=f'끝 (turn {int(tt[-1])})')
            rep_rot = '반시계' if _signed_area(x, y) > 0 else '시계'
        ax.set_xlim(0, xmax * 1.06)
        ax.set_ylim(0, ymax * 1.22)    # 위쪽 여백: 회전 방향 설명 상자가 궤적을 가리지 않게
        ax.set_xlabel(f'{_name(prey)} 수 (마리, 피식자) — {LV_SMOOTH} step 이동평균', color=_color(prey))
        ax.set_ylabel(f'{_name(pred)} 수 (마리, 포식자) — {LV_SMOOTH} step 이동평균', color=_color(pred))
        ax.set_title(f'{_name(pred)} – {_name(prey)}', loc='left')
        if len(runs) > 1:
            ax.plot([], [], color='#c8c8c8', lw=1.5, label='다른 seed')
        ax.legend(loc='upper right', frameon=False, fontsize=11)
        lines = [f'seed {seed} 궤적의 회전: {rep_rot} 방향',
                 f'반시계로 도는 seed: {ccw}/{n_alive}',
                 f'(부호 면적, turn {LV_BURN_IN} 이후 · 멸종 전까지)']
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, ha='left', va='top',
                fontsize=11, color=INK,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', alpha=0.9))
    if lc is not None:
        cb = fig.colorbar(lc, ax=list(axes), pad=0.02, fraction=0.03 if len(pairs) > 1 else 0.05)
        cb.set_label(f'turn (seed {seed}, 진할수록 나중)')
        cb.outline.set_visible(False)
    fig.suptitle(f'로트카-볼테라 위상도 — {_pname(policy)} (seed {seed} 강조, 외부 유입 끔) · '
                 f'반시계 고리 = 피식자가 늘면 포식자가 뒤따라 늘고, 피식자가 줄면 뒤따라 준다',
                 x=0.06, ha='left', fontweight='bold', fontsize=14)
    _save(fig, out_dir, f'lv_phase_{policy}.png', saved)


def plot_lv_summary(rows, out_dir, saved):
    """lv_summary.csv: 정책 · 쌍별 지연/주기 · 주기 · 교차상관 최대값(점 = seed, 굵은 가로선 = 평균)."""
    if not rows:
        return
    pols = [p for p in LV_POLICIES if any(str(r.get('policy')) == p for r in rows)]
    pols += sorted({str(r.get('policy')) for r in rows} - set(pols))
    pair_keys = []
    for r in rows:
        k = (str(r.get('predator')), str(r.get('prey')))
        if k not in pair_keys:
            pair_keys.append(k)
    groups = [(p, pr) for pr in pair_keys for p in pols]      # x 위치: 쌍마다 정책끼리 붙임
    xs = []
    x = 0.0
    for gi, (p, pr) in enumerate(groups):
        if gi and groups[gi - 1][1] != pr:
            x += 0.6        # 쌍 사이 간격
        xs.append(x)
        x += 1.0
    marker = {p: m for p, m in zip(pols, ('o', 's', 'D', '^'))}
    panels = (('lag_ratio', '포식자 지연 ÷ 주기', True),
              ('period', '주기 (step)', True),
              ('xcorr_max', '교차상관 최대값 r', False))
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), gridspec_kw={'wspace': 0.3})
    for ax, (key, label, only_measurable) in zip(axes, panels):
        for gx, (p, (pred, prey)) in zip(xs, groups):
            sel = [r for r in rows if str(r.get('policy')) == p and str(r.get('predator')) == pred
                   and str(r.get('prey')) == prey]
            meas = [r for r in sel if int(_fnum(r.get('measurable')) or 0) == 1]
            src = meas if only_measurable else sel
            vals = [(v, int(_fnum(r.get('measurable')) or 0) == 1)
                    for r in src for v in (_fnum(r.get(key)),) if v is not None]
            col = _color(pred)
            for i, (v, m) in enumerate(vals):
                jitter = (i - (len(vals) - 1) / 2) * 0.07
                ax.plot(gx + jitter, v, marker[p], ms=9, color=col if m else 'white', mec=col,
                        mew=1.8, zorder=3)
            if vals:
                mean = statistics.fmean(v for v, _ in vals)
                ax.plot([gx - 0.3, gx + 0.3], [mean, mean], color=INK, lw=2.4, zorder=4)
                fmt = f'{mean:.2f}' if key != 'period' else f'{mean:.0f}'
                ax.annotate(fmt, (gx + 0.3, mean), xytext=(3, 0), textcoords='offset points',
                            va='center', fontsize=10, color=INK)
            ax.annotate(f'측정 가능\n{len(meas)}/{len(sel)}', (gx, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -38), textcoords='offset points', ha='center', va='top',
                        fontsize=10, color=MUTED)
        if key == 'lag_ratio':
            ax.axhline(LV_THEORY, color='#2b8a3e', lw=1.6, ls='--', zorder=2)
            # 이름표는 두 쌍 사이 빈 칸 위에(점과 겹치지 않게)
            gaps = [(xs[i - 1] + xs[i]) / 2 for i in range(1, len(groups))
                    if groups[i - 1][1] != groups[i][1]]
            ax.annotate(f'LV 이론\n{LV_THEORY:g}', (gaps[0] if gaps else xs[-1] + 0.5, LV_THEORY),
                        xytext=(0, -6), textcoords='offset points', ha='center', va='top',
                        fontsize=11,
                        color='#2b8a3e', fontweight='bold')
            ax.axhline(0, color='#9a9a9a', lw=0.8)
        if key == 'xcorr_max':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_xticks(xs, [f'{POLICY_NAME.get(p, p)}\n{_name(pr[0])}–{_name(pr[1])}'
                           for p, pr in groups], fontsize=11)
        ax.set_title(label, loc='left')
        ax.grid(axis='x', visible=False)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.8)
    handles = [plt.Line2D([], [], marker=marker[p], ls='', color='#555555', ms=9,
                          label=POLICY_NAME.get(p, p)) for p in pols]
    handles += [plt.Line2D([], [], marker='o', ls='', color='white', mec='#555555', mew=1.8, ms=9,
                           label='빈 점 = 주기 측정 불가 seed'),
                plt.Line2D([], [], color=INK, lw=2.4, label='가로선 = 평균')]
    handles += [plt.Line2D([], [], marker='s', ls='', color=_color(pr[0]), ms=10,
                           label=f'{_name(pr[0])}–{_name(pr[1])}') for pr in pair_keys]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.06, 0.93), ncol=len(handles),
               frameon=False, fontsize=11)
    steps = {r.get('steps') for r in rows}
    fig.suptitle(f'로트카-볼테라 유사성 요약 (lv_summary.csv · seed별 점 · '
                 f'{"/".join(str(s) for s in steps)} step · 외부 유입 끔 · 처음 {LV_BURN_IN} step 제외)',
                 x=0.06, ha='left', fontweight='bold', fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    _save(fig, out_dir, 'lv_summary.png', saved)


# ---------------------------------------------------------------- 진입점
def _run_jobs(jobs, out_dir, saved):
    for fn, args, kw in jobs:   # 한 차트가 실패해도 나머지는 계속 만든다
        try:
            fn(*args, out_dir, saved, **kw)
        except Exception:
            plt.close('all')
            print(f'[plots] {fn.__name__}{tuple(a if isinstance(a, str) else "…" for a in args)} 실패:')
            traceback.print_exc()


def _eval_jobs(results_dir, with_summary=True):
    """results_dir 의 eval_<policy>.csv 로 만들 평가 차트 작업 목록."""
    evals = {}
    for p in POLICIES:
        rows = _read_csv(os.path.join(results_dir, f'eval_{p}.csv'))
        if rows and species_from_columns(rows[0].keys()):
            evals[p] = _by_seed(rows)
    jobs = []
    for p, runs in evals.items():
        jobs += [(plot_population, (p, runs), {}), (plot_oscillation, (p, runs), {}),
                 (plot_phase, (p, runs), {}), (plot_needs, (p, runs), {})]
    if evals:
        summary_rows = (_read_csv(os.path.join(results_dir, 'eval_summary.csv'))
                        if with_summary else None)
        jobs.append((plot_compare, (evals,), {'summary_rows': summary_rows}))
    return jobs


def make_all(results_dir: str = 'data/results') -> list[str]:
    """있는 CSV로 만들 수 있는 차트를 모두 만들고 저장한 PNG 경로 목록을 돌려준다."""
    _setup_style()
    out_dir = os.path.join(results_dir, 'plots')
    saved: list[str] = []

    jobs = _eval_jobs(results_dir)
    train = _read_csv(os.path.join(results_dir, 'train_log.csv'))
    if train:
        jobs.append((plot_training, (train,), {}))

    # LV(로트카-볼테라) 차트
    lv_summary = _read_csv(os.path.join(results_dir, 'lv_summary.csv'))
    summ = _lv_summary_index(lv_summary)
    for p in LV_POLICIES:
        rows = _read_csv(os.path.join(results_dir, f'lv_{p}.csv'))
        if rows and species_from_columns(rows[0].keys()):
            runs = _by_seed(rows)
            jobs += [(plot_lv_timeseries, (p, runs, summ), {}), (plot_lv_seeds, (p, runs, summ), {}),
                     (plot_lv_phase, (p, runs, summ), {})]
    if lv_summary:
        jobs.append((plot_lv_summary, (lv_summary,), {}))

    # 외부 유입을 끈 평가(noimm/eval_*.csv) → plots/noimm/
    noimm_dir = os.path.join(results_dir, 'noimm')
    noimm_jobs = _eval_jobs(noimm_dir, with_summary=False) if os.path.isdir(noimm_dir) else []

    if not jobs and not noimm_jobs:
        return saved
    if jobs:
        os.makedirs(out_dir, exist_ok=True)
        _run_jobs(jobs, out_dir, saved)
    if noimm_jobs:
        sub = os.path.join(out_dir, 'noimm')
        os.makedirs(sub, exist_ok=True)
        _NOTE['title'] = ' · 외부 유입 끔'
        try:
            _run_jobs(noimm_jobs, sub, saved)
        finally:
            _NOTE['title'] = ''
    return saved


if __name__ == '__main__':
    import sys
    for p in make_all(sys.argv[1] if len(sys.argv) > 1 else 'data/results'):
        print(p)
