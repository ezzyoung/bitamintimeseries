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

- **모델 구조 실험 1: BiLSTM + Attention**

![image](https://github.com/user-attachments/assets/de5ea29c-a82d-4443-b1f1-ae8bdb9b07ff)

- **모델 구조 실험 2 :Cross Attention with BiGRU**

![image](https://github.com/user-attachments/assets/1aa79b7c-9b26-460a-8415-3095e1f2dc1e)

- **모델 구조 실험 3 :CNN + TCN + Attention**

![image](https://github.com/user-attachments/assets/e7d28355-29c7-41ea-9fb3-d9f918c2e16e)

- **모델 구조 실험 4 (최종 선택) : 5 CNN + Linear Classifer**


- **image_text** 

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
