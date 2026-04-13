# Pair Profiler Representative Report

- representative datasets: qm9_hf (small), mptrj (large)
- cases: baseline, full_legacy, full_no_expand
- modes: force_model, forward_energy

이 파일들은 non-Flash 경로에서 연산이 어떻게 쪼개지는지 보기 위한 `torch.profiler` 결과다.
headline latency 비교가 아니라, 연산 분해와 호출 수 확인용이다.
