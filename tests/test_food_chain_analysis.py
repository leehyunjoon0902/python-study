"""LV 유사성 측정 도구(food_chain.analysis) 테스트: 합성 파형으로 주기 · 지연 · 진폭을 확인한다."""
import csv
import math
import tempfile
import unittest
from pathlib import Path
from random import Random

from food_chain import analysis, config


def wave(n, period, lag=0.0, mean=100.0, amp=50.0, noise=0.0, seed=0):
    rng = Random(seed)
    return [max(1, round(mean + amp * math.sin(2 * math.pi * (t - lag) / period)
                         + rng.gauss(0, noise))) for t in range(n)]


class SignalTest(unittest.TestCase):
    def test_moving_average_keeps_length_and_flattens_noise(self):
        x = [0, 10] * 50
        y = analysis.moving_average(x, 11)
        self.assertEqual(len(y), len(x))
        self.assertTrue(all(4.0 <= v <= 6.0 for v in y[5:-5]))
        self.assertEqual(analysis.moving_average([1, 2, 3], 1), [1.0, 2.0, 3.0])

    def test_detrend_removes_line(self):
        y = analysis.detrend([3 + 2 * i for i in range(50)])
        self.assertTrue(all(abs(v) < 1e-9 for v in y))

    def test_autocorrelation_peak_of_sine(self):
        x = [math.sin(2 * math.pi * t / 50) for t in range(500)]
        acf = analysis.autocorrelation(x, 200)
        self.assertAlmostEqual(acf[0], 1.0)
        self.assertEqual(analysis.first_acf_peak(acf), 50)
        self.assertEqual(analysis.autocorrelation([5.0] * 20, 5), [0.0] * 6)

    def test_no_peak_for_monotone_series(self):
        x = [float(t) for t in range(300)]
        self.assertIsNone(analysis.first_acf_peak(analysis.autocorrelation(x, 150)))

    def test_best_lag_sign(self):
        prey = [math.sin(2 * math.pi * t / 80) for t in range(800)]
        pred = [math.sin(2 * math.pi * (t - 20) / 80) for t in range(800)]
        lag, r = analysis.best_lag(prey, pred, 40)
        self.assertEqual(lag, 20)                 # 포식자가 20 늦다 → 양수
        self.assertGreater(r, 0.99)
        lag, _ = analysis.best_lag(pred, prey, 40)
        self.assertEqual(lag, -20)


class PairTest(unittest.TestCase):
    def test_synthetic_lv_wave(self):
        """주기 500, 포식자 지연 125(주기의 0.25), 피식자 진폭 150/50 = 3배."""
        n = 8001
        prey = wave(n, 500, 0, 100, 50, noise=4, seed=1)
        pred = wave(n, 500, 125, 20, 10, noise=1, seed=2)
        r = analysis.analyze_pair(prey, pred)
        self.assertEqual(r["measurable"], 1)
        self.assertAlmostEqual(r["period"], 500, delta=20)
        self.assertAlmostEqual(r["period_predator"], 500, delta=20)
        self.assertAlmostEqual(r["period_peaks"], 500, delta=25)
        self.assertAlmostEqual(r["lag"], 125, delta=16)
        self.assertAlmostEqual(r["lag_ratio"], 0.25, delta=0.04)
        self.assertGreater(r["xcorr_max"], 0.9)
        self.assertAlmostEqual(r["cycles"], 14.0, delta=0.6)       # (8001 − 1000) / 500
        self.assertAlmostEqual(r["amplitude_prey"], 3.0, delta=0.4)
        self.assertAlmostEqual(r["amplitude_predator"], 3.0, delta=0.5)
        self.assertEqual(r["extinct_step"], -1)

    def test_fewer_than_two_cycles_is_not_measurable(self):
        n = 8001
        prey = wave(n, 5000, 0)
        pred = wave(n, 5000, 1250, 20, 10)
        r = analysis.analyze_pair(prey, pred)
        self.assertEqual(r["measurable"], 0)

    def test_flat_noise_has_no_amplitude(self):
        rng = Random(3)
        prey = [100 + rng.randrange(-3, 4) for _ in range(6001)]
        pred = [20 + rng.randrange(-1, 2) for _ in range(6001)]
        r = analysis.analyze_pair(prey, pred)
        self.assertLess(r["amplitude_prey"], 1.1)
        self.assertLess(abs(r["xcorr_max"]), 0.6)

    def test_extinction_truncates_analysis(self):
        prey = wave(8001, 500)
        pred = wave(8001, 500, 125, 20, 10)
        pred[4000:] = [0] * (8001 - 4000)
        r = analysis.analyze_pair(prey, pred)
        self.assertEqual(r["extinct_step"], 4000)
        self.assertEqual(r["analyzed_steps"], 3000)
        self.assertAlmostEqual(r["period"], 500, delta=20)
        pred[1200:] = [0] * (8001 - 1200)                    # 과도 구간 직후 멸종: 구간이 너무 짧다
        r = analysis.analyze_pair(prey, pred)
        self.assertEqual(r["measurable"], 0)
        self.assertIsNone(r["period"])

    def test_predation_share_counts_only_decline_steps(self):
        n = 6001
        prey = wave(n, 500)
        pred = wave(n, 500, 125, 20, 10)
        deaths = [4] * n
        killed = [0] * n
        sm = analysis.moving_average(prey, analysis.SMOOTH_WINDOW)
        for t in range(1, n):
            if sm[t] < sm[t - 1]:
                killed[t] = 3                                # 줄어드는 step에서만 포식 3/4
        r = analysis.analyze_pair(prey, pred, deaths, killed)
        self.assertAlmostEqual(r["predation_share"], 0.75, delta=0.01)
        self.assertGreater(r["decline_steps"], 2000)


class RunTest(unittest.TestCase):
    def test_run_lv_writes_csvs_and_restores_immigration(self):
        saved = config.IMMIGRATION_ENABLED
        with tempfile.TemporaryDirectory() as d:
            lines = []
            rows = analysis.run_lv("rule", seeds=1, steps=60, first_seed=5, results_dir=d,
                                   out=lines.append)
            self.assertEqual(config.IMMIGRATION_ENABLED, saved)
            self.assertEqual([(r["predator"], r["prey"]) for r in rows], list(analysis.PAIRS))
            self.assertTrue(all(r["measurable"] == 0 for r in rows))     # 60 step: 측정 불가
            self.assertTrue(any("측정 불가" in x for x in lines))
            with open(Path(d) / "lv_rule.csv", encoding="utf-8") as f:
                data = list(csv.reader(f))
            self.assertEqual(data[0], ["seed"] + analysis.series_keys())
            self.assertEqual(data[0][:6], ["seed", "turn"] + [f"count_{s}" for s in config.SPECIES])
            self.assertEqual(len(data), 1 + 61)
            with open(Path(d) / "lv_summary.csv", encoding="utf-8") as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 2)
            self.assertEqual(tuple(summary[0]), analysis.SUMMARY_COLUMNS)
            # 다른 policy를 추가로 돌리면 기존 policy 행은 남는다
            analysis.run_lv("random", seeds=1, steps=20, first_seed=5, results_dir=d, out=None)
            with open(Path(d) / "lv_summary.csv", encoding="utf-8") as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(sorted({r["policy"] for r in summary}), ["random", "rule"])

    def test_series_kills_are_per_step(self):
        rows = analysis.run_series("rule", 7, 80)
        self.assertEqual(rows[0]["turn"], 0)
        self.assertEqual(rows[-1]["turn"], 80)
        self.assertTrue(all(r["kills_fox_rabbit"] <= r["deaths_rabbit"] for r in rows))
        self.assertGreater(sum(r["kills_fox_rabbit"] for r in rows), 0)


if __name__ == "__main__":
    unittest.main()
