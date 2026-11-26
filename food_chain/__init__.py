"""먹이사슬 생태계 강화학습: 토끼 · 사슴(초식, SARSA) + 여우 · 늑대(육식, Q-Learning).

장치: 토끼 굴(은신), 사슴 질주(직선 2칸 이동), 늑대 2칸 사냥 반경, 늑대 영역성 · 기근 대사,
      먹이 분리(토끼 = 풀, 사슴 = 관목), 에너지 예산 번식(번식 비용 = 새끼의 시작 energy).

    python -m food_chain --train     # 학습 → data/q_tables/, data/results/train_log.csv
    python -m food_chain --eval      # random / rule / trained 비교
    python -m food_chain --lv --policy rule   # 로트카-볼테라 유사성(주기 · 포식자 지연 · 진폭) 측정
    python -m food_chain --watch     # 뷰어
    python -m food_chain --plots     # 차트

코어(config, world, rl, train, simulation, analysis)는 표준 라이브러리만 사용한다.
"""
