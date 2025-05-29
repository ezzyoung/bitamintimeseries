# XAI Project : 의료 수면데이터 중심으로

## 🌟 Project Topic : Explainable AI (XAI) 관점에서의 수면 데이터 

- 기존에 Black Box 문제가 있던 의료 딥러닝 모델 XAI 분석
- 해석가능성이 딥러닝보다 높은 머신러닝 모델 사용 및 의료 도메인 연관 XAI 분석

## 🌟 Project Objective

- 모델 성능
- 해석 가능성
- 경량 및 단순화

## ⚙️ **Process**

### 🧩 `Detailed Process'
- **수면 단계 예측**
    - 딥러닝 기반 블랙박스 모델 예측 및 XAI
    - 머신러닝 기반 예측 및 XAI
- **수면 장애 탐지**
    - 딥러닝 기반 장애 탐지 및 XAI
    - 머신러닝 기반 장애 탐지 및 XAI
- **수면 패턴 클러스터링**


### 🧩 `Flowchart`

![image](https://github.com/user-attachments/assets/2ab21a7d-fb10-460c-a7d1-e5d734b279e2)

### 📊 `Data Collection`
**Sleep-EDF Dataset**

**ISRUC-Sleep-EDF Dataset**

### 🧪 `Experiment`


#### A. Sleep Stage Prediction - DL

- Dataset Merge : 120 명의 Sleep-EDF Data PSG + Hypnogram 합쳐서 pt 형식으로 조합 후 사용
- 구조 변경 참고 모델 : **TCN, SleePyCo, Cross-Modal Transformers**

- **모델 구조 실험 1: BiLSTM + Attention**

![image](https://github.com/user-attachments/assets/de5ea29c-a82d-4443-b1f1-ae8bdb9b07ff)

    - EEG/EOG 각각의 신호를 CNN으로 전처리하여 특징 추출
    - 두 모달리티를 결합한 후 BiLSTM으로 시계열 정보를 학습
    - Attention 메커니즘을 통해 중요한 시간 구간에 집중
    - Classifier를 통해 최종 수면 단계 예측

- **모델 구조 실험 2 :Cross Attention with BiGRU**

![image](https://github.com/user-attachments/assets/1aa79b7c-9b26-460a-8415-3095e1f2dc1e)

    - EEG/EOG 각각에서 저수준~고수준 특징을 CNN으로 추출
    - Q/K/V 기반 Cross-Modal Attention으로 두 모달리티 간 상호작용 학습
    - BiGRU를 통해 시간 흐름을 반영한 순차 정보 처리
    - Fully Connected Layer로 수면 단계 분류

- **모델 구조 실험 3 :CNN + TCN + Attention**

![image](https://github.com/user-attachments/assets/e7d28355-29c7-41ea-9fb3-d9f918c2e16e)

    - 위 Cross Attention with BiGRU 에서 GRU 대신 TCN 사용
    - Attention Pooling 적용
    - Query-Based Attention Pooling 적용도 실험

- **모델 구조 실험 4 (최종 선택) : 5 CNN + Linear Classifer**
  
  - 5개의 CNN 블록, 13개의 1D CNN, 각 블록마다 SE 블록 추가하여 Attention 추가

- **최종 모델 성능**

![image](https://github.com/user-attachments/assets/9a3c8e7d-2f72-47e2-b45b-786a693c207d)

- **weight** 

- **input images**  


#### **Result**

#### B. 분석

- **prompt** 

- **image_text** 

- **weight** 

- **input images**  


#### **Result**


## 🎯 **Usefulness of the Project**
- 블랙박스 딥러닝 기반으로도 수면 단계 예측 가능
- 머신러닝으로 딥러닝 만능주의에서 벗어나 좋은 성능 달성 : 해석가능성 및 경량화
- 헬스케어의 신뢰도 이슈 머신러닝& 딥러닝 통해 해결 가능 → 경제성 고려 가능함

## 📂 **Project Information**

### **🧑‍🤝‍🧑 Team Members**

| 기수  | 팀원 |
|------|------|
| **15기** | 김가원, 박신지, 김범준, 이지영, 김나연|



### **📅 Progress Period**

- 2025.03.03 ~ 2025.05.28

 


### **📌 Repository Structure**  
```bash
📂 BITAmin-TimeSeries
│── 📂 SleepStage_DL/               # 딥러닝 기반 수면단계 예측 및 분석 실험
│── 📂 SleepStage_ML/         # 머신러닝 기반 수면단계 예측 및 분석 실험
│── 📂 SleepAnomaly_DL/          # 딥러닝 기반 수면장애 예측 및 분석 실험
│── 📂 SleepAnomaly_ML/      # 머신러닝 기반 수면장애 예측 및 분석 실험
│── 📂 SleepPattern/      # 수면패턴 클러스터링 실험
│── README.md             # 프로젝트 개요 및 진행 내용
