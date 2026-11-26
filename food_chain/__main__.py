"""CLI: python -m food_chain --train | --eval | --lv | --watch | --plots"""
import argparse
import sys

from . import config


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m food_chain",
                                description="먹이사슬 생태계 강화학습 (초식 토끼·사슴 = SARSA, 육식 여우·늑대 = Q-Learning)")
    p.add_argument("--train", action="store_true", help="학습 후 Q-Table · 학습 로그 저장")
    p.add_argument("--episodes", type=int, default=None, help="학습 episode 수")
    p.add_argument("--train-steps", type=int, default=None, help="학습 episode 길이")
    p.add_argument("--schedule", choices=("alternating", "joint"), default="alternating",
                   help="교대 학습(기본) / 동시 학습")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--eval", action="store_true", help="random / rule / trained 비교")
    p.add_argument("--eval-ablation", action="store_true",
                   help="--eval에 '포식자 없음' 대조 실행을 추가(단독으로 써도 평가를 실행)")
    p.add_argument("--lv", action="store_true",
                   help="로트카-볼테라 유사성 측정(주기 · 포식자 지연 · 진폭). --policy rule|trained, "
                        "기본 seed 4개 × 8000 step, 외부 유입 끔")
    p.add_argument("--seeds", type=int, default=None,
                   help=f"평가 seed 수(--eval 기본 {config.EVAL_SEEDS}, --lv 기본 4)")
    p.add_argument("--steps", type=int, default=None,
                   help=f"평가 step 수(--eval 기본 {config.EVAL_STEPS}, --lv 기본 8000)")
    p.add_argument("--no-immigration", action="store_true",
                   help="--eval을 외부 유입 없이 실행(--lv는 원래 유입을 끈다)")
    p.add_argument("--immigration", action="store_true", help="--lv를 외부 유입을 켠 채 실행")
    p.add_argument("--results-dir", default=None, help="--eval / --lv CSV를 저장할 폴더")
    p.add_argument("--watch", action="store_true", help="뷰어 실행")
    p.add_argument("--policy", choices=("trained", "random", "rule"), default="trained")
    p.add_argument("--seed", type=int, default=42, help="뷰어 배치 seed")
    p.add_argument("--plots", action="store_true", help="차트 PNG 저장")
    args = p.parse_args(argv)

    if args.eval_ablation:
        args.eval = True
    if not (args.train or args.eval or args.lv or args.watch or args.plots):
        p.print_help()
        return 0

    for stream in (sys.stdout, sys.stderr):          # Windows 콘솔에서도 한글이 깨지지 않게
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if args.train:
        from .train import train
        train(episodes=args.episodes, steps=args.train_steps, seed=args.train_seed,
              schedule=args.schedule)
    if args.eval:
        from .simulation import evaluate
        if args.no_immigration:
            config.IMMIGRATION_ENABLED = False
        evaluate(seeds=args.seeds, steps=args.steps, ablation=args.eval_ablation,
                 results_dir=args.results_dir)
    if args.lv:
        from .analysis import run_lv
        try:
            run_lv(policy=args.policy, seeds=args.seeds, steps=args.steps,
                   immigration=args.immigration, results_dir=args.results_dir)
        except FileNotFoundError as err:
            print(err)
            return 1
    if args.plots:
        from . import plots                           # 지연 import (matplotlib 필요)
        for path in plots.make_all(results_dir=str(config.RESULTS_DIR)):
            print(path)
    if args.watch:
        from .simulation import Simulation
        from . import viewer                          # 지연 import (tkinter 필요)
        try:
            sim = Simulation(seed=args.seed, policy_mode=args.policy)
        except FileNotFoundError as err:
            print(err)
            return 1
        viewer.run(sim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
