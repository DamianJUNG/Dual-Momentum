# 임시 해결책: 직접 라이브러리 설치
import subprocess
import sys

def install_requirements():
    """필요한 라이브러리들을 직접 설치"""
    packages = [
        'streamlit==1.28.1',
        'pandas==1.5.3', 
        'numpy==1.24.3',
        'yfinance==0.2.18',
        'plotly==5.17.0',
        'scipy==1.10.1'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except:
            pass  # 이미 설치된 경우 무시

# 라이브러리 설치 실행
install_requirements()

# 기존 import들
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
from scipy import stats
import io
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
from scipy import stats
import io
import base64

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Quant-Runner | 동적 자산 배분 백테스팅",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    
    .performance-positive {
        color: #28a745;
    }
    
    .performance-negative {
        color: #dc3545;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import modules (they will be defined in the same file for simplicity)
@dataclass
class StrategyConfig:
    """전략별 설정 클래스"""
    name: str = 'RelativeStrengthMomentum'
    lookback_period_months: int = 5
    num_assets_to_hold: int = 2
    rebalance_frequency: str = 'monthly'
    
@dataclass 
class RiskManagementConfig:
    """리스크 관리 설정"""
    enable_stop_loss: bool = False
    stop_loss_threshold: float = -0.10
    enable_volatility_filter: bool = False
    max_portfolio_volatility: float = 0.20
    cash_asset: str = 'SHY'

@dataclass
class TransactionConfig:
    """거래비용 설정"""
    commission_rate: float = 0.001
    bid_ask_spread: float = 0.0005
    enable_transaction_costs: bool = True

@dataclass
class Configuration:
    """전체 백테스트 설정"""
    universe_tickers: List[str] = field(default_factory=lambda: [
        'SHY', 'TLT', 'RWR', 'IWM', 'IWB', 'GLD', 'EFA', 'EEM', 'DBC'
    ])
    benchmark_ticker: str = 'SPY'
    start_date: str = '2007-01-01'
    end_date: str = '2023-12-31'
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    transaction: TransactionConfig = field(default_factory=TransactionConfig)

class DataFetcher:
    """금융 데이터 수집 및 전처리"""
    
    def __init__(self, config: Configuration):
        self.config = config
        
    @st.cache_data(ttl=3600)  # 1시간 캐시
    def fetch_data(_self) -> pd.DataFrame:
        """모든 필요한 데이터 수집"""
        try:
            extended_start = _self._calculate_extended_start_date()
            
            all_tickers = list(set(
                _self.config.universe_tickers + 
                [_self.config.benchmark_ticker] +
                [_self.config.risk_management.cash_asset]
            ))
            
            with st.spinner(f'📥 데이터 다운로드 중... ({len(all_tickers)}개 자산)'):
                data = yf.download(
                    all_tickers,
                    start=extended_start,
                    end=_self.config.end_date,
                    progress=False
                )['Adj Close']
                
            if isinstance(data, pd.Series):
                data = data.to_frame(all_tickers[0])
            
            initial_rows = len(data)
            data = data.dropna()
            dropped_rows = initial_rows - len(data)
            
            if dropped_rows > 0:
                st.info(f"📊 결측치로 인해 {dropped_rows}개 행이 제거되었습니다.")
            
            if data.empty:
                raise ValueError("유효한 데이터가 없습니다.")
                
            st.success(f"✅ 데이터 수집 완료: {len(data)}개 일자, {len(data.columns)}개 자산")
            return data
            
        except Exception as e:
            st.error(f"❌ 데이터 수집 오류: {e}")
            raise
    
    def _calculate_extended_start_date(self) -> str:
        """룩백 기간을 고려한 시작일 계산"""
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        months_to_subtract = self.config.strategy.lookback_period_months + 3
        extended_start = start_date - timedelta(days=months_to_subtract * 31)
        return extended_start.strftime('%Y-%m-%d')

class BacktestingEngine:
    """핵심 백테스팅 로직"""
    
    def __init__(self, config: Configuration, price_data: pd.DataFrame):
        self.config = config
        self.price_data = price_data
        
    def run_backtest(self) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """백테스트 실행"""
        monthly_prices = self._resample_to_monthly(self.price_data)
        monthly_returns = monthly_prices.pct_change().dropna()
        
        backtest_start = pd.to_datetime(self.config.start_date)
        monthly_returns = monthly_returns[monthly_returns.index >= backtest_start]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        portfolio_returns = []
        positions_history = []
        detailed_history = []
        
        total_months = len(monthly_returns)
        
        for i, current_date in enumerate(monthly_returns.index):
            progress = (i + 1) / total_months
            progress_bar.progress(progress)
            status_text.text(f'백테스트 진행: {i+1}/{total_months} ({progress*100:.1f}%)')
            
            if i == 0:
                selected_assets = self.config.universe_tickers[:self.config.strategy.num_assets_to_hold]
                asset_returns = monthly_returns.loc[current_date, selected_assets]
                portfolio_return = asset_returns.mean()
                momentum_scores = {asset: 0 for asset in selected_assets}
            else:
                selected_assets, momentum_scores = self._select_assets_by_momentum_detailed(
                    monthly_returns, current_date, i
                )
                
                if self.config.risk_management.enable_stop_loss:
                    selected_assets = self._apply_stop_loss_filter(
                        selected_assets, monthly_returns, current_date
                    )
                
                if selected_assets:
                    asset_returns = monthly_returns.loc[current_date, selected_assets]
                    portfolio_return = asset_returns.mean()
                    
                    if self.config.transaction.enable_transaction_costs:
                        transaction_cost = self._calculate_transaction_costs(
                            selected_assets, positions_history[-1] if positions_history else []
                        )
                        portfolio_return -= transaction_cost
                else:
                    cash_asset = self.config.risk_management.cash_asset
                    portfolio_return = monthly_returns.loc[current_date, cash_asset]
                    asset_returns = pd.Series({cash_asset: portfolio_return})
                    selected_assets = [cash_asset]
            
            detail_record = {
                'date': current_date,
                'portfolio_return': portfolio_return,
                'selected_assets': selected_assets,
                'asset_weights': {asset: 1/len(selected_assets) for asset in selected_assets},
                'individual_returns': asset_returns.to_dict() if hasattr(asset_returns, 'to_dict') else asset_returns,
                'momentum_scores': momentum_scores
            }
            detailed_history.append(detail_record)
            
            portfolio_returns.append(portfolio_return)
            positions_history.append(selected_assets)
        
        progress_bar.progress(1.0)
        status_text.text('✅ 백테스트 완료!')
        
        portfolio_series = pd.Series(portfolio_returns, index=monthly_returns.index)
        positions_df = pd.DataFrame(positions_history, index=monthly_returns.index)
        detailed_df = pd.DataFrame(detailed_history)
        
        return portfolio_series, positions_df, detailed_df
    
    def _resample_to_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.resample('M').last()
    
    def _select_assets_by_momentum_detailed(self, returns_data: pd.DataFrame, 
                                           current_date: pd.Timestamp, period_index: int) -> Tuple[List[str], Dict[str, float]]:
        lookback_periods = self.config.strategy.lookback_period_months
        
        if period_index < lookback_periods:
            selected = self.config.universe_tickers[:self.config.strategy.num_assets_to_hold]
            momentum_scores = {asset: 0 for asset in selected}
            return selected, momentum_scores
        
        end_idx = period_index
        start_idx = end_idx - lookback_periods
        
        momentum_returns = returns_data.iloc[start_idx:end_idx][self.config.universe_tickers]
        cumulative_returns = (1 + momentum_returns).prod() - 1
        
        ranked_assets = cumulative_returns.sort_values(ascending=False)
        selected_assets = ranked_assets.head(self.config.strategy.num_assets_to_hold).index.tolist()
        momentum_scores = cumulative_returns.to_dict()
        
        return selected_assets, momentum_scores
    
    def _apply_stop_loss_filter(self, selected_assets: List[str], 
                               returns_data: pd.DataFrame, current_date: pd.Timestamp) -> List[str]:
        threshold = self.config.risk_management.stop_loss_threshold
        current_returns = returns_data.loc[current_date, selected_assets]
        
        filtered_assets = [asset for asset in selected_assets 
                          if current_returns[asset] > threshold]
        
        return filtered_assets if filtered_assets else [self.config.risk_management.cash_asset]
    
    def _calculate_transaction_costs(self, new_positions: List[str], 
                                   old_positions: List[str]) -> float:
        if not old_positions:
            num_trades = len(new_positions)
        else:
            position_changes = set(new_positions) ^ set(old_positions)
            num_trades = len(position_changes)
        
        cost_per_trade = (self.config.transaction.commission_rate + 
                         self.config.transaction.bid_ask_spread)
        total_cost = cost_per_trade * (num_trades / len(self.config.universe_tickers))
        
        return total_cost

class PerformanceAnalyzer:
    """성과 지표 계산 및 분석"""
    
    def __init__(self, config: Configuration):
        self.config = config
        
    def analyze_performance(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict[str, Any]:
        results = {
            'portfolio_metrics': self._calculate_metrics(portfolio_returns),
            'benchmark_metrics': self._calculate_metrics(benchmark_returns),
            'relative_metrics': self._calculate_relative_metrics(portfolio_returns, benchmark_returns),
            'risk_metrics': self._calculate_risk_metrics(portfolio_returns, benchmark_returns),
            'period_analysis': self._analyze_by_periods(portfolio_returns, benchmark_returns)
        }
        
        return results
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        years = len(returns) / 12
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(12)
        sharpe_ratio = cagr / volatility if volatility != 0 else 0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
        sortino_ratio = cagr / downside_vol if downside_vol != 0 else 0
        
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'best_month': returns.max(),
            'worst_month': returns.min()
        }
    
    def _calculate_relative_metrics(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> Dict[str, float]:
        if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
            beta, alpha, r_value, p_value, std_err = stats.linregress(
                benchmark_returns, portfolio_returns
            )
            correlation = portfolio_returns.corr(benchmark_returns)
        else:
            beta, alpha, correlation = 0, 0, 0
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(12)
        information_ratio = active_returns.mean() * 12 / tracking_error if tracking_error != 0 else 0
        
        up_capture = self._calculate_capture_ratio(portfolio_returns, benchmark_returns, 'up')
        down_capture = self._calculate_capture_ratio(portfolio_returns, benchmark_returns, 'down')
        
        return {
            'alpha': alpha * 12,
            'beta': beta,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture
        }
    
    def _calculate_capture_ratio(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series, direction: str) -> float:
        if direction == 'up':
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
        
        if mask.sum() == 0:
            return 0
        
        portfolio_avg = portfolio_returns[mask].mean()
        benchmark_avg = benchmark_returns[mask].mean()
        
        return portfolio_avg / benchmark_avg if benchmark_avg != 0 else 0
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series) -> Dict[str, float]:
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _analyze_by_periods(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict[str, Any]:
        portfolio_annual = portfolio_returns.groupby(portfolio_returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        benchmark_annual = benchmark_returns.groupby(benchmark_returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        portfolio_monthly = portfolio_returns.groupby(portfolio_returns.index.month).mean()
        benchmark_monthly = benchmark_returns.groupby(benchmark_returns.index.month).mean()
        
        return {
            'annual_returns': {
                'portfolio': portfolio_annual.to_dict(),
                'benchmark': benchmark_annual.to_dict()
            },
            'monthly_seasonality': {
                'portfolio': portfolio_monthly.to_dict(),
                'benchmark': benchmark_monthly.to_dict()
            }
        }

def create_equity_curve_chart(portfolio_returns: pd.Series, benchmark_returns: pd.Series, benchmark_name: str):
    """누적 수익률 곡선 차트"""
    port_cumret = (1 + portfolio_returns).cumprod()
    bench_cumret = (1 + benchmark_returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=port_cumret.index,
        y=port_cumret.values,
        mode='lines',
        name='포트폴리오',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='%{x}<br>누적수익률: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=bench_cumret.index,
        y=bench_cumret.values,
        mode='lines',
        name=f'벤치마크 ({benchmark_name})',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='%{x}<br>누적수익률: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '📈 누적 수익률 곡선',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='날짜',
        yaxis_title='누적 수익률',
        hovermode='x unified',
        legend=dict(x=0, y=1),
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_drawdown_chart(portfolio_returns: pd.Series, benchmark_returns: pd.Series, benchmark_name: str):
    """낙폭 곡선 차트"""
    def calculate_drawdowns(returns):
        cumret = (1 + returns).cumprod()
        rolling_max = cumret.expanding().max()
        return (cumret - rolling_max) / rolling_max
    
    port_dd = calculate_drawdowns(portfolio_returns)
    bench_dd = calculate_drawdowns(benchmark_returns)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=port_dd.index,
        y=port_dd.values * 100,
        mode='lines',
        name='포트폴리오',
        fill='tonexty',
        line=dict(color='#d62728', width=2),
        hovertemplate='%{x}<br>낙폭: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=bench_dd.index,
        y=bench_dd.values * 100,
        mode='lines',
        name=f'벤치마크 ({benchmark_name})',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='%{x}<br>낙폭: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '📉 낙폭(Drawdown) 곡선',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='날짜',
        yaxis_title='낙폭 (%)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_annual_returns_chart(performance_results: Dict[str, Any]):
    """연도별 수익률 차트"""
    annual_data = performance_results['period_analysis']['annual_returns']
    years = sorted(annual_data['portfolio'].keys())
    
    port_returns = [annual_data['portfolio'][year] * 100 for year in years]
    bench_returns = [annual_data['benchmark'][year] * 100 for year in years]
    
    fig = go.Figure(data=[
        go.Bar(name='포트폴리오', x=years, y=port_returns, 
               marker_color='#1f77b4', 
               hovertemplate='%{x}<br>수익률: %{y:.2f}%<extra></extra>'),
        go.Bar(name='벤치마크', x=years, y=bench_returns, 
               marker_color='#ff7f0e',
               hovertemplate='%{x}<br>수익률: %{y:.2f}%<extra></extra>')
    ])
    
    fig.update_layout(
        title={
            'text': '📊 연도별 수익률 비교',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='연도',
        yaxis_title='수익률 (%)',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_monthly_selections_chart(detailed_df: pd.DataFrame, config: Configuration):
    """매월 선택 종목 차트"""
    recent_data = detailed_df.tail(24)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('월별 포트폴리오 수익률', '선택된 자산 (색상: 개별 수익률)'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.6]
    )
    
    # 월별 수익률
    colors = ['green' if x > 0 else 'red' for x in recent_data['portfolio_return']]
    fig.add_trace(
        go.Bar(
            x=[d.strftime('%Y-%m') for d in recent_data['date']],
            y=recent_data['portfolio_return'] * 100,
            name='포트폴리오 수익률 (%)',
            marker_color=colors
        ),
        row=1, col=1
    )
    
    # 선택된 자산 히트맵
    assets = config.universe_tickers
    selection_matrix = np.zeros((len(assets), len(recent_data)))
    
    for j, (_, row) in enumerate(recent_data.iterrows()):
        for asset in row['selected_assets']:
            if asset in assets:
                asset_idx = assets.index(asset)
                individual_ret = row['individual_returns'].get(asset, 0)
                selection_matrix[asset_idx, j] = individual_ret * 100
    
    fig.add_trace(
        go.Heatmap(
            z=selection_matrix,
            x=[d.strftime('%Y-%m') for d in recent_data['date']],
            y=assets,
            colorscale='RdYlGn',
            colorbar=dict(title="개별 수익률 (%)"),
            zmid=0,
            name='자산 선택'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title={
            'text': '🎯 월별 선택 종목 및 수익률',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Quant-Runner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">동적 자산 배분 백테스팅 플랫폼</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ 백테스트 설정")
        
        # 기본 설정
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📅 기간 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "시작일",
                value=datetime(2007, 1, 1),
                min_value=datetime(2000, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "종료일",
                value=datetime(2023, 12, 31),
                min_value=datetime(2000, 1, 1),
                max_value=datetime.now()
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 자산 유니버스
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 자산 유니버스")
        
        preset_universes = {
            "기본 (글로벌 분산)": ['SHY', 'TLT', 'RWR', 'IWM', 'IWB', 'GLD', 'EFA', 'EEM', 'DBC'],
            "보수적": ['SHY', 'TLT', 'IWB', 'GLD', 'EFA'],
            "공격적": ['QQQ', 'IWM', 'EEM', 'GLD', 'TLT', 'RWR', 'DBC'],
            "미국 중심": ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ'],
            "섹터 로테이션": ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
        }
        
        universe_preset = st.selectbox(
            "프리셋 선택",
            options=list(preset_universes.keys()),
            index=0
        )
        
        universe_tickers = st.multiselect(
            "자산 티커 (편집 가능)",
            options=['SPY', 'QQQ', 'IWM', 'IWB', 'TLT', 'SHY', 'GLD', 'SLV', 'VNQ', 'RWR', 
                    'EFA', 'EEM', 'DBC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU'],
            default=preset_universes[universe_preset]
        )
        
        benchmark_ticker = st.selectbox(
            "벤치마크",
            options=['SPY', 'QQQ', 'IWB', 'VTI', 'VT'],
            index=0
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 전략 설정
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🧠 전략 설정")
        
        lookback_months = st.slider(
            "모멘텀 룩백 기간 (월)",
            min_value=1,
            max_value=12,
            value=5,
            help="과거 몇 개월의 수익률을 기반으로 모멘텀을 계산할지 설정"
        )
        
        num_assets = st.slider(
            "보유 자산 수",
            min_value=1,
            max_value=min(10, len(universe_tickers)),
            value=min(2, len(universe_tickers)),
            help="상위 몇 개 자산을 보유할지 설정"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 리스크 관리
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚠️ 리스크 관리")
        
        enable_stop_loss = st.checkbox(
            "스톱로스 활성화",
            value=False,
            help="개별 자산의 손실이 임계값을 초과하면 현금으로 이동"
        )
        
        if enable_stop_loss:
            stop_loss_threshold = st.slider(
                "스톱로스 임계값 (%)",
                min_value=-20.0,
                max_value=-1.0,
                value=-10.0,
                step=0.5
            ) / 100
        else:
            stop_loss_threshold = -0.10
        
        enable_transaction_costs = st.checkbox(
            "거래비용 반영",
            value=True,
            help="수수료와 스프레드를 반영하여 더 현실적인 백테스트"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 백테스트 실행 버튼
        run_backtest = st.button(
            "🚀 백테스트 실행",
            type="primary",
            use_container_width=True
        )
    
    # Main content
    if run_backtest:
        if len(universe_tickers) < num_assets:
            st.error("⚠️ 보유 자산 수가 유니버스 크기보다 클 수 없습니다!")
            return
        
        # Configuration 생성
        config = Configuration(
            universe_tickers=universe_tickers,
            benchmark_ticker=benchmark_ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            strategy=StrategyConfig(
                lookback_period_months=lookback_months,
                num_assets_to_hold=num_assets
            ),
            risk_management=RiskManagementConfig(
                enable_stop_loss=enable_stop_loss,
                stop_loss_threshold=stop_loss_threshold
            ),
            transaction=TransactionConfig(
                enable_transaction_costs=enable_transaction_costs
            )
        )
        
        try:
            # 데이터 수집
            fetcher = DataFetcher(config)
            price_data = fetcher.fetch_data()
            
            # 백테스트 실행
            engine = BacktestingEngine(config, price_data)
            portfolio_returns, positions_df, detailed_df = engine.run_backtest()
            
            # 벤치마크 수익률
            benchmark_prices = price_data[benchmark_ticker].resample('M').last()
            benchmark_returns = benchmark_prices.pct_change().dropna()
            backtest_start = pd.to_datetime(config.start_date)
            benchmark_returns = benchmark_returns[benchmark_returns.index >= backtest_start]
            
            # 성과 분석
            analyzer = PerformanceAnalyzer(config)
            performance_results = analyzer.analyze_performance(portfolio_returns, benchmark_returns)
            
            # 결과 표시
            st.success("✅ 백테스트 완료!")
            
            # 성과 요약 카드
            st.markdown("## 📊 성과 요약")
            
            port_metrics = performance_results['portfolio_metrics']
            bench_metrics = performance_results['benchmark_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cagr_diff = port_metrics['cagr'] - bench_metrics['cagr']
                color_class = "performance-positive" if cagr_diff > 0 else "performance-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">연복리수익률 (CAGR)</div>
                    <div class="metric-value {color_class}">{port_metrics['cagr']*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #666;">
                        벤치마크: {bench_metrics['cagr']*100:.2f}% 
                        ({cagr_diff*100:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                vol_diff = port_metrics['volatility'] - bench_metrics['volatility']
                color_class = "performance-positive" if vol_diff < 0 else "performance-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">연변동성</div>
                    <div class="metric-value {color_class}">{port_metrics['volatility']*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #666;">
                        벤치마크: {bench_metrics['volatility']*100:.2f}% 
                        ({vol_diff*100:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                sharpe_diff = port_metrics['sharpe_ratio'] - bench_metrics['sharpe_ratio']
                color_class = "performance-positive" if sharpe_diff > 0 else "performance-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">샤프 지수</div>
                    <div class="metric-value {color_class}">{port_metrics['sharpe_ratio']:.3f}</div>
                    <div style="font-size: 0.8rem; color: #666;">
                        벤치마크: {bench_metrics['sharpe_ratio']:.3f} 
                        ({sharpe_diff:+.3f})
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                mdd_diff = port_metrics['max_drawdown'] - bench_metrics['max_drawdown']
                color_class = "performance-positive" if mdd_diff > 0 else "performance-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">최대낙폭 (MDD)</div>
                    <div class="metric-value {color_class}">{port_metrics['max_drawdown']*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #666;">
                        벤치마크: {bench_metrics['max_drawdown']*100:.2f}% 
                        ({mdd_diff*100:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 탭 구성
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 수익률 곡선", "📉 리스크 분석", "🎯 포지션 분석", "📊 상세 지표", "📋 데이터 다운로드"
            ])
            
            with tab1:
                st.plotly_chart(
                    create_equity_curve_chart(portfolio_returns, benchmark_returns, benchmark_ticker),
                    use_container_width=True
                )
                
                st.plotly_chart(
                    create_drawdown_chart(portfolio_returns, benchmark_returns, benchmark_ticker),
                    use_container_width=True
                )
                
                st.plotly_chart(
                    create_annual_returns_chart(performance_results),
                    use_container_width=True
                )
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_metrics = performance_results['risk_metrics']
                    rel_metrics = performance_results['relative_metrics']
                    
                    st.markdown("### 📊 리스크 지표")
                    risk_data = {
                        "지표": ["VaR (95%)", "CVaR (95%)", "스큐니스", "첨도", "베타", "상관계수"],
                        "값": [
                            f"{risk_metrics['var_95']*100:.2f}%",
                            f"{risk_metrics['cvar_95']*100:.2f}%",
                            f"{risk_metrics['skewness']:.3f}",
                            f"{risk_metrics['kurtosis']:.3f}",
                            f"{rel_metrics['beta']:.3f}",
                            f"{rel_metrics['correlation']:.3f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
                
                with col2:
                    st.markdown("### 📈 상대 성과")
                    relative_data = {
                        "지표": ["알파", "추적오차", "정보비율", "상승포착률", "하락포착률"],
                        "값": [
                            f"{rel_metrics['alpha']*100:.2f}%",
                            f"{rel_metrics['tracking_error']*100:.2f}%",
                            f"{rel_metrics['information_ratio']:.3f}",
                            f"{rel_metrics['up_capture_ratio']:.3f}",
                            f"{rel_metrics['down_capture_ratio']:.3f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(relative_data), use_container_width=True)
            
            with tab3:
                st.plotly_chart(
                    create_monthly_selections_chart(detailed_df, config),
                    use_container_width=True
                )
                
                # 자산별 선택 빈도
                st.markdown("### 🎯 자산별 선택 통계")
                
                all_selections = []
                for _, row in detailed_df.iterrows():
                    all_selections.extend(row['selected_assets'])
                
                selection_counts = pd.Series(all_selections).value_counts()
                total_months = len(detailed_df)
                
                asset_stats = []
                for asset in config.universe_tickers:
                    count = selection_counts.get(asset, 0)
                    frequency = count / total_months
                    
                    momentum_scores = []
                    for _, row in detailed_df.iterrows():
                        if asset in row['momentum_scores']:
                            momentum_scores.append(row['momentum_scores'][asset])
                    
                    avg_momentum = np.mean(momentum_scores) if momentum_scores else 0
                    
                    asset_stats.append({
                        "자산": asset,
                        "선택횟수": count,
                        "선택비율": f"{frequency*100:.1f}%",
                        "평균모멘텀": f"{avg_momentum*100:.2f}%"
                    })
                
                asset_stats_df = pd.DataFrame(asset_stats)
                st.dataframe(asset_stats_df, use_container_width=True)
            
            with tab4:
                # 연도별 상세 수익률
                st.markdown("### 📅 연도별 수익률 상세")
                annual_data = performance_results['period_analysis']['annual_returns']
                
                annual_comparison = []
                for year in sorted(annual_data['portfolio'].keys()):
                    port_ret = annual_data['portfolio'][year]
                    bench_ret = annual_data['benchmark'][year]
                    excess = port_ret - bench_ret
                    
                    annual_comparison.append({
                        "연도": year,
                        "포트폴리오": f"{port_ret*100:.2f}%",
                        "벤치마크": f"{bench_ret*100:.2f}%",
                        "초과수익": f"{excess*100:+.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(annual_comparison), use_container_width=True)
                
                # 최근 선택 이력
                st.markdown("### 🗓️ 최근 12개월 선택 이력")
                recent_selections = detailed_df.tail(12).copy()
                recent_selections['날짜'] = recent_selections['date'].dt.strftime('%Y-%m')
                recent_selections['선택자산'] = recent_selections['selected_assets'].apply(lambda x: ', '.join(x))
                recent_selections['포트폴리오수익률'] = recent_selections['portfolio_return'].apply(lambda x: f"{x*100:+.2f}%")
                
                display_df = recent_selections[['날짜', '선택자산', '포트폴리오수익률']]
                st.dataframe(display_df, use_container_width=True)
            
            with tab5:
                st.markdown("### 📥 결과 다운로드")
                
                # 포트폴리오 수익률 CSV
                portfolio_csv = portfolio_returns.to_csv()
                st.download_button(
                    label="📊 포트폴리오 수익률 다운로드 (CSV)",
                    data=portfolio_csv,
                    file_name=f"portfolio_returns_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
                # 상세 이력 CSV
                detailed_csv = detailed_df.to_csv(index=False)
                st.download_button(
                    label="📋 상세 이력 다운로드 (CSV)",
                    data=detailed_csv,
                    file_name=f"detailed_history_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
                # 성과 요약 JSON
                import json
                performance_json = json.dumps(performance_results, indent=2, default=str)
                st.download_button(
                    label="📈 성과 지표 다운로드 (JSON)",
                    data=performance_json,
                    file_name=f"performance_metrics_{start_date}_{end_date}.json",
                    mime="application/json"
                )
                
                st.markdown("""
                <div class="info-box">
                    <strong>💡 활용 팁:</strong><br>
                    • 포트폴리오 수익률: 다른 도구와 성과 비교<br>
                    • 상세 이력: 매월 선택 종목 및 모멘텀 스코어 분석<br>
                    • 성과 지표: 종합적인 리스크-수익률 분석
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ 백테스트 실행 중 오류가 발생했습니다: {str(e)}")
            st.markdown("""
            <div class="warning-box">
                <strong>🔍 문제 해결 방법:</strong><br>
                • 인터넷 연결 확인<br>
                • 유효한 티커 심볼 확인<br>
                • 날짜 범위 조정<br>
                • 페이지 새로고침 후 재시도
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # 초기 화면
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Quant-Runner란?</h3>
            <p>Quant-Runner는 동적 자산 배분 전략의 백테스팅을 위한 전문 플랫폼입니다.</p>
            
            <h4>✨ 주요 기능</h4>
            <ul>
                <li><strong>모멘텀 기반 전략:</strong> 상대강도를 활용한 자산 선택</li>
                <li><strong>리스크 관리:</strong> 스톱로스, 거래비용 반영</li>
                <li><strong>상세 분석:</strong> 30+ 성과 지표 및 리스크 메트릭</li>
                <li><strong>시각화:</strong> 인터랙티브 차트 및 그래프</li>
                <li><strong>데이터 내보내기:</strong> CSV, JSON 형태 다운로드</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>🚀 시작하기</h4>
            <ol>
                <li>왼쪽 사이드바에서 백테스트 설정을 조정하세요</li>
                <li>자산 유니버스와 전략 파라미터를 선택하세요</li>
                <li>'백테스트 실행' 버튼을 클릭하세요</li>
                <li>결과를 분석하고 다운로드하세요</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # 샘플 프리셋 설명
        st.markdown("### 📋 프리셋 설명")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🛡️ 보수적 전략**
            - 국채, 대형주, 금 중심
            - 낮은 변동성, 안정적 수익
            - 은퇴자 및 보수적 투자자 적합
            """)
            
            st.markdown("""
            **🌍 기본 (글로벌 분산)**
            - 전 세계 자산군 포함
            - 주식, 채권, 원자재, 리츠
            - 균형잡힌 포트폴리오
            """)
        
        with col2:
            st.markdown("""
            **🚀 공격적 전략**
            - 성장주, 신흥국 중심
            - 높은 변동성, 높은 수익 추구
            - 젊은 투자자에게 적합
            """)
            
            st.markdown("""
            **🏢 섹터 로테이션**
            - 미국 섹터 ETF 중심
            - 경기 사이클에 따른 섹터 순환
            - 전술적 자산배분 전략
            """)

if __name__ == "__main__":
    main()
