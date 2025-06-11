# Dual-Momentum
Dual Momentum Asset Allocation
# 📊 Quant-Runner

<div align="center">

![Quant-Runner Logo](https://via.placeholder.com/400x150/1f77b4/ffffff?text=Quant-Runner)

**동적 자산 배분 백테스팅 플랫폼**

Portfolio Visualizer와 같은 전문적인 백테스팅 기능을 제공하는 웹 기반 플랫폼

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/quant-runner?style=social)](https://github.com/your-username/quant-runner)

[🚀 라이브 데모](https://your-app-name.streamlit.app) •
[📖 문서](https://github.com/your-username/quant-runner/wiki) •
[🐛 버그 신고](https://github.com/your-username/quant-runner/issues) •
[💡 기능 요청](https://github.com/your-username/quant-runner/discussions)

</div>

---

## 🎯 개요

Quant-Runner는 **모멘텀 기반 전술적 자산 배분(Tactical Asset Allocation)** 전략을 구현하고 백테스트할 수 있는 무료 오픈소스 플랫폼입니다. Portfolio Visualizer의 유료 기능들을 무료로 제공하며, 한국 투자자들에게 특화된 기능을 포함합니다.

### 🔥 주요 특징

- **📈 완전한 백테스팅 엔진**: 모멘텀 기반 동적 자산 선택
- **🎨 세련된 웹 인터페이스**: Streamlit 기반 반응형 디자인  
- **📊 30+ 성과 지표**: CAGR, Sharpe, VaR, CVaR 등 전문 지표
- **🔒 리스크 관리**: 스톱로스, 변동성 필터링, 거래비용 반영
- **🌍 글로벌 자산**: 주식, 채권, 원자재, 리츠, 국제 자산 지원
- **💾 데이터 내보내기**: CSV, JSON 형태로 결과 다운로드
- **🆓 완전 무료**: 제한 없는 백테스트 및 모든 기능 이용

---

## 🚀 빠른 시작

### 📱 온라인에서 바로 사용

**별도 설치 없이 웹 브라우저에서 바로 이용하세요!**

👉 **[Quant-Runner 웹앱 실행](https://your-app-name.streamlit.app)** 👈

### 💻 로컬 설치

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/quant-runner.git
cd quant-runner

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 앱 실행
streamlit run app.py
```

🎉 **브라우저에서 `http://localhost:8501` 접속!**

---

## 🎬 스크린샷

<div align="center">

### 🏠 메인 대시보드
![Main Dashboard](https://via.placeholder.com/800x500/f8f9fa/333333?text=Main+Dashboard+Screenshot)

### 📈 수익률 곡선 분석
![Performance Chart](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Equity+Curve+Analysis)

### 🎯 포지션 분석
![Position Analysis](https://via.placeholder.com/800x400/ff7f0e/ffffff?text=Position+Heatmap)

</div>

---

## ✨ 주요 기능

### 🧠 백테스팅 엔진

<table>
<tr>
<td>

**📊 전략 기능**
- 상대강도 모멘텀 기반 자산 선택
- 1~12개월 룩백 기간 설정
- 상위 N개 자산 동적 선택
- 월별 자동 리밸런싱

</td>
<td>

**⚠️ 리스크 관리**  
- 스톱로스 임계값 설정
- 변동성 기반 필터링
- 현금 대체 자산 설정
- 거래비용 및 슬리피지 반영

</td>
</tr>
</table>

### 📈 성과 분석

<details>
<summary><strong>🔍 30+ 전문 성과 지표 (클릭하여 펼치기)</strong></summary>

**📊 수익률 지표**
- CAGR (연복리수익률)
- 총 수익률  
- 연도별/월별 수익률

**⚡ 리스크 지표**
- 연변동성 (표준편차)
- 최대낙폭 (MDD) 
- VaR/CVaR (95%, 99%)
- 스큐니스, 첨도

**🎯 위험조정수익률**
- 샤프 지수 (Sharpe Ratio)
- 소르티노 지수 (Sortino Ratio)  
- 칼마 지수 (Calmar Ratio)

**📋 벤치마크 대비**
- 알파 (Alpha)
- 베타 (Beta)
- 추적오차 (Tracking Error)
- 정보비율 (Information Ratio)
- 상승/하락 포착률

</details>

### 🎨 사용자 인터페이스

- **🎛️ 직관적 설정**: 사이드바에서 모든 파라미터 조정
- **📱 반응형 디자인**: 모바일, 태블릿, 데스크톱 완벽 지원
- **🎨 세련된 테마**: 전문적이고 신뢰감 있# 📊 Quant-Runner

**동적 자산 배분 백테스팅 플랫폼**

Quant-Runner는 Portfolio Visualizer와 같은 전문적인 백테스팅 기능을 제공하는 웹 기반 플랫폼입니다. 모멘텀 기반 전술적 자산 배분(Tactical Asset Allocation) 전략을 구현하고 백테스트할 수 있습니다.

## 🚀 라이브 데모

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## ✨ 주요 기능

### 📈 백테스팅 엔진
- **모멘텀 기반 전략**: 상대강도를 활용한 동적 자산 선택
- **리스크 관리**: 스톱로스, 변동성 필터링
- **거래비용 반영**: 수수료, 스프레드, 슬리피지 고려
- **다양한 자산군**: 주식, 채권, 원자재, 리츠, 국제 자산

### 📊 성과 분석
- **30+ 성과 지표**: CAGR, Sharpe, Sortino, Calmar, VaR, CVaR 등
- **벤치마크 비교**: 알파, 베타, 추적오차, 정보비율
- **리스크 분석**: 최대낙폭, 하락변동성, 스큐니스, 첨도
- **기간별 분석**: 연도별, 월별 성과 분해

### 📱 사용자 인터페이스
- **직관적 설정**: 드래그 앤 드롭으로 간편한 파라미터 조정
- **인터랙티브 차트**: Plotly 기반 고품질 시각화
- **반응형 디자인**: 모바일, 태블릿, 데스크톱 지원
- **데이터 내보내기**: CSV, JSON 형태 다운로드

## 🛠️ 설치 및 실행

### 로컬 실행

1. **저장소 클론**
```bash
git clone https://github.com/your-username/quant-runner.git
cd quant-runner
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

3. **앱 실행**
```bash
streamlit run app.py
```

4. **브라우저에서 접속**
```
http://localhost:8501
```

### Streamlit Cloud 배포

1. GitHub 저장소 생성 및 코드 업로드
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 배포
3. 자동으로 웹앱 URL 생성

## 📋 사용법

### 1. 기본 설정
- **기간 설정**: 백테스트 시작일과 종료일 선택
- **자산 유니버스**: 투자할 자산들의 티커 심볼 입력
- **벤치마크**: 성과 비교 기준 설정

### 2. 전략 파라미터
- **모멘텀 룩백 기간**: 1~12개월 선택 (기본: 5개월)
- **보유 자산 수**: 상위 몇 개 자산을 보유할지 설정
- **리밸런싱 주기**: 월별 리밸런싱 (주별 지원 예정)

### 3. 리스크 관리
- **스톱로스**: 개별 자산 손실 한계 설정
- **거래비용**: 현실적인 거래비용 반영
- **현금 대체**: 조건 미충족 시 안전자산으로 대피

### 4. 결과 분석
- **수익률 곡선**: 포트폴리오 vs 벤치마크 누적 성과
- **리스크 분석**: 낙폭, VaR, 상관관계 등 리스크 지표
- **포지션 분석**: 매월 선택 종목 및 기여도 분석
- **상세 지표**: 연도별 성과, 선택 빈도 등 세부 분석

## 🎯 프리셋 전략

### 보수적 전략
```python
universe: ['SHY', 'TLT', 'IWB', 'GLD', 'EFA']
lookback: 6개월
보유자산: 3개
```

### 기본 (글로벌 분산)
```python
universe: ['SHY', 'TLT', 'RWR', 'IWM', 'IWB', 'GLD', 'EFA', 'EEM', 'DBC']
lookback: 5개월
보유자산: 2개
```

### 공격적 전략
```python
universe: ['QQQ', 'IWM', 'EEM', 'GLD', 'TLT', 'RWR', 'DBC']
lookback: 3개월
보유자산: 2개
```

### 섹터 로테이션
```python
universe: ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
lookback: 3개월
보유자산: 3개
```

## 📊 지원 자산군

### 미국 주식
- **SPY**: S&P 500
- **QQQ**: 나스닥 100
- **IWM**: Russell 2000 (소형주)
- **IWB**: Russell 1000 (대형주)

### 채권
- **TLT**: 20년+ 미국 국채
- **SHY**: 1-3년 미국 국채
- **AGG**: 종합 채권

### 원자재 & 금
- **GLD**: 금
- **SLV**: 은
- **DBC**: 원자재 바스켓

### 부동산
- **VNQ**: 미국 리츠
- **RWR**: 글로벌 리츠

### 국제주식
- **EFA**: 선진국 (유럽, 호주, 극동)
- **EEM**: 신흥국
- **VEA**: 선진국 (FTSE)

### 섹터 ETF
- **XLY**: 소비재
- **XLE**: 에너지
- **XLF**: 금융
- **XLK**: 기술
- **XLV**: 헬스케어
- 기타 9개 섹터

## 🔧 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python
- **데이터**: yfinance (Yahoo Finance API)
- **분석**: pandas, numpy, scipy
- **시각화**: Plotly
- **배포**: Streamlit Cloud

## 📈 성과 지표

### 수익률 지표
- **CAGR**: 연복리수익률
- **총수익률**: 전체 기간 누적 수익률
- **연도별 수익률**: 각 연도별 성과

### 리스크 지표
- **변동성**: 연율화 표준편차
- **최대낙폭 (MDD)**: 최고점 대비 최대 하락률
- **VaR/CVaR**: 95%, 99% 신뢰구간 손실
- **스큐니스/첨도**: 수익률 분포 특성

### 위험조정수익률
- **샤프 지수**: 단위 변동성당 초과수익률
- **소르티노 지수**: 하락변동성 기준 조정수익률
- **칼마 지수**: 최대낙폭 기준 조정수익률

### 벤치마크 대비
- **알파**: 벤치마크 대비 초과수익률
- **베타**: 시장 민감도
- **추적오차**: 벤치마크와의 수익률 차이 변동성
- **정보비율**: 추적오차 대비 초과수익률

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ 면책조항

이 도구는 교육 및 연구 목적으로 제작되었습니다. 투자 결정에 참고용으로만 사용하시기 바라며, 모든 투자 책임은 사용자에게 있습니다. 과거 성과가 미래 수익을 보장하지 않습니다.

## 📞 문의

- **개발자**: [Your Name]
- **이메일**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)

## 🙏 감사의 말

- **Portfolio Visualizer**: 영감을 제공한 훌륭한 플랫폼
- **Meb Faber**: 동적 자산 배분 연구의 선구자
- **Streamlit**: 빠른 웹앱 개발을 가능하게 한 프레임워크

---

⭐ 이 프로젝트가 도움이 되셨다면 스타를 눌러주세요!
