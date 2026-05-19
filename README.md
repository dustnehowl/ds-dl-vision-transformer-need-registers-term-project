### 발표 논문 : Vision Transfomers need registers
## 목차 : 
  1. Overview
     (1) 논문에서 제기하는 문제
     (2) 문제 해결을 위한 제안 방법
  2. Problem Formulration in this paper
     (1) 실험에 사용한 알고리즘 : Deit-III, OpenCLIP, DINOv3, DINOv1
     (2) 실험 결과 : without rester vs with register image 
  3. Re-Production experiments
     (1) 우리가 실험하고자 하는 방안 : Deit-III, OpenCLIP, DINOv3, DINOv1 구현해서 실험 결과와 동일한지 비교(2번이랑 합칠지는 생각해 보고)
     (2) 알고리즘 소스 확보 방안 : DINOv1 은 어디서, DINOv3는 어디서 ....... 등등
     (3) 실험 환경
         - 학습 데이터
         - 인프라, GPU
     (4) 실험 결과
         - 각 알고리즘에 대한 결과 비교. artifact가 사라졌는지 아닌지. 비판적 비교
  4. Conclusion
  5. Contribution
