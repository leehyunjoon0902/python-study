"""먹이사슬 시뮬레이션 뷰어 실행 파일.

VS Code에서 이 파일을 열고 오른쪽 위의 ▶(Run Python File) 버튼을 누르면 뷰어가 뜬다.
명령어로는 `python run_viewer.py`와 같다.

아래 두 값만 바꾸면 된다.
"""
from food_chain.__main__ import main

# 'rule'    = 규칙 기반 정책(사람이 짠 규칙, 포식자–피식자 진동이 크게 보인다)
# 'trained' = 학습된 정책(강화학습으로 채운 Q-Table, data/q_tables 필요)
# 'random'  = 무작위 정책(비교용, 곧 멸종한다)
POLICY = "rule"

# 동물 · 지형의 초기 배치를 정하는 번호. 바꾸면 다른 지도에서 시작한다.
SEED = 42

if __name__ == "__main__":
    main(["--watch", "--policy", POLICY, "--seed", str(SEED)])
