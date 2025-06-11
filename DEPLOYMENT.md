# 🚀 Quant-Runner 배포 가이드

이 가이드는 Quant-Runner를 Streamlit Cloud를 통해 무료로 배포하는 방법을 설명합니다.

## 📁 파일 구조

배포를 위해 다음과 같은 파일 구조가 필요합니다:

```
quant-runner/
├── app.py                 # 메인 Streamlit 앱
├── requirements.txt       # Python 의존성
├── README.md             # 프로젝트 설명
├── DEPLOYMENT.md         # 배포 가이드 (이 파일)
├── .gitignore           # Git 무시 파일
└── .streamlit/
    └── config.toml      # Streamlit 설정
```

## 🔧 1단계: GitHub 저장소 생성

### 1.1 GitHub에서 새 저장소 생성
1. [GitHub](https://github.com)에 로그인
2. 우상단 '+' 버튼 → 'New repository' 클릭
3. Repository name: `quant-runner` (또는 원하는 이름)
4. Description: `동적 자산 배분 백테스팅 플랫폼`
5. Public으로 설정 (Streamlit Cloud 무료 버전은 Public 저장소만 지원)
6. 'Create repository' 클릭

### 1.2 로컬에서 저장소 설정
```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/quant-runner.git
cd quant-runner

# 파일들 추가 (위의 모든 파일들을 복사)
# app.py, requirements.txt, README.md, .gitignore, .streamlit/config.toml

# Git 설정
git add .
git commit -m "Initial commit: Quant-Runner 백테스팅 플랫폼"
git push origin main
```

## 🌐 2단계: Streamlit Cloud 배포

### 2.1 Streamlit Cloud 계정 생성
1. [Streamlit Cloud](https://streamlit.io/cloud)에 접속
2. 'Sign up' 클릭
3. GitHub 계정으로 로그인 (권장)

### 2.2 앱 배포
1. Streamlit Cloud 대시보드에서 'New app' 클릭
2. 다음 정보 입력:
   ```
   Repository: YOUR_USERNAME/quant-runner
   Branch: main
   Main file path: app.py
   App URL: quant-runner (또는 원하는 이름)
   ```
3. 'Deploy!' 클릭

### 2.3 배포 완료
- 약 2-5분 후 배포 완료
- 자동으로 생성된 URL: `https://quant-runner.streamlit.app`
- GitHub 저장소에 push할 때마다 자동으로 재배포

## 🔄 3단계: 업데이트 및 유지보수

### 3.1 코드 업데이트
```bash
# 코드 수정 후
git add .
git commit -m "기능 개선: 새로운 전략 추가"
git push origin main
```
- GitHub에 push하면 Streamlit Cloud에서 자동으로 재배포됩니다.

### 3.2 라이브러리 추가
새로운 Python 라이브러리가 필요한 경우:
1. `requirements.txt`에 라이브러리 추가
2. GitHub에 push
3. Streamlit Cloud에서 자동으로 새 라이브러리 설치

### 3.3 설정 변경
Streamlit 설정을 변경하려면:
1. `.streamlit/config.toml` 파일 수정
2. GitHub에 push
3. 자동으로 새 설정 적용

## 🎨 4단계: 커스터마이징

### 4.1 도메인 커스터마이징
Streamlit Cloud 무료 버전에서는 커스텀 도메인을 직접 지원하지 않지만, 다음과 같은 방법들이 있습니다:

1. **GitHub Pages를 통한 리다이렉트**
2. **Cloudflare를 통한 프록시** (고급 사용자)
3. **Heroku, AWS 등 다른 플랫폼 사용** (유료)

### 4.2 테마 및 스타일링
`.streamlit/config.toml`에서 테마 커스터마이징:
```toml
[theme]
primaryColor = "#FF6B6B"           # 메인 색상
backgroundColor = "#FFFFFF"        # 배경색
secondaryBackgroundColor = "#F0F0F0"  # 사이드바 색상
textColor = "#262730"             # 텍스트 색상
```

### 4.3 로고 및 파비콘
```python
# app.py에 추가
st.set_page_config(
    page_title="Your App Name",
    page_icon="🚀",  # 또는 이미지 경로
    layout="wide"
)
```

## 📊 5단계: 성능 최적화

### 5.1 캐싱 활용
데이터 다운로드 성능 향상을 위해 `@st.cache_data` 사용:
```python
@st.cache_data(ttl=3600)  # 1시간 캐시
def fetch_data():
    # 데이터 다운로드 로직
    pass
```

### 5.2 메모리 사용량 최적화
- 큰 DataFrame은 적절히 필터링
- 불필요한 글로벌 변수 제거
- 이미지는 압축하여 사용

### 5.3 로딩 시간 단축
```python
# 프로그레스 바 표시
progress_bar = st.progress(0)
status_text = st.empty()

for i, item in enumerate(items):
    progress_bar.progress((i + 1) / len(items))
    status_text.text(f'처리 중: {i+1}/{len(items)}')
    # 작업 수행
```

## 🔒 6단계: 보안 및 비밀 정보

### 6.1 Secrets 관리
API 키 등 민감한 정보는 Streamlit Secrets를 사용:

1. Streamlit Cloud 앱 대시보드에서 'Settings' 클릭
2. 'Secrets' 탭에서 설정:
   ```toml
   [api_keys]
   alpha_vantage = "YOUR_API_KEY"
   quandl = "YOUR_QUANDL_KEY"
   ```

3. 코드에서 사용:
   ```python
   import streamlit as st
   api_key = st.secrets["api_keys"]["alpha_vantage"]
   ```

### 6.2 사용량 제한
무료 버전 제한 사항:
- **CPU**: 1 vCPU
- **RAM**: 800MB
- **동시 접속자**: 제한 없음 (하지만 성능 영향)
- **배포 앱 수**: 3개

## 🔄 7단계: 대안 배포 방법

### 7.1 Heroku 배포
```bash
# Procfile 생성
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile