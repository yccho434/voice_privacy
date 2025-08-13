# audio_enhancer.py
"""
음성 품질 향상 모듈 (Audio Enhancement Module)
경진대회용 전처리 파이프라인
- 노이즈 감소
- 스펙트럴 필터링
- 음성 대역 최적화
- 시각화 및 지표 측정
"""

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Dict, Optional
import os
from datetime import datetime
import platform

# matplotlib 한글 폰트 설정
import matplotlib.font_manager as fm
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'  # 맥
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 리눅스
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class AudioEnhancer:
    """음성 품질 향상 클래스"""
    
    def __init__(self, target_sr: int = 16000):
        """
        Args:
            target_sr: 목표 샘플링 레이트 (ETRI STT 권장: 16000)
        """
        self.target_sr = target_sr
        self.metrics = {}
        
    def enhance(self, audio_path: str, output_path: Optional[str] = None, 
                visualize: bool = True) -> Tuple[str, Dict]:
        """
        음성 품질 향상 메인 파이프라인
        
        Args:
            audio_path: 입력 음성 파일 경로
            output_path: 출력 파일 경로 (None이면 자동 생성)
            visualize: 시각화 여부
            
        Returns:
            (출력 파일 경로, 개선 지표 딕셔너리)
        """
        print("="*50)
        print("🎵 음성 품질 향상 파이프라인 시작")
        print("="*50)
        
        # 1. 음성 로드
        print("📂 음성 파일 로드 중...")
        y_original, sr_original = librosa.load(audio_path, sr=None)
        y, sr = librosa.load(audio_path, sr=self.target_sr)
        print(f"  ✓ 원본 샘플레이트: {sr_original}Hz → {self.target_sr}Hz 리샘플링")
        
        # 원본 지표 측정
        metrics_before = self._calculate_metrics(y, sr, "원본")
        
        # 2. 노이즈 프로파일 분석
        print("\n🔍 노이즈 프로파일 분석 중...")
        noise_profile = self._analyze_noise_profile(y, sr)
        print(f"  ✓ 노이즈 레벨: {noise_profile['noise_level']:.2f} dB")
        print(f"  ✓ SNR: {noise_profile['snr']:.2f} dB")
        
        # 3. 스펙트럴 노이즈 게이팅
        print("\n🎚️ 스펙트럴 노이즈 게이팅 적용 중...")
        y_denoised = self._spectral_gating(y, sr, noise_profile)
        print("  ✓ 주파수 도메인 노이즈 제거 완료")
        
        # 4. 음성 대역 강조 (300-3400Hz)
        print("\n📻 음성 주파수 대역 최적화 중...")
        y_enhanced = self._enhance_speech_band(y_denoised, sr)
        print("  ✓ 음성 명료도 향상 필터 적용")
        
        # 5. 다이나믹 레인지 정규화
        print("\n📊 다이나믹 레인지 정규화 중...")
        y_normalized = self._normalize_dynamic_range(y_enhanced)
        print("  ✓ 음량 균일화 완료")
        
        # 처리 후 지표 측정
        metrics_after = self._calculate_metrics(y_normalized, sr, "처리 후")
        
        # 개선율 계산 및 metrics 저장 (중요: 시각화 전에!)
        improvement = self._calculate_improvement(metrics_before, metrics_after)
        self.metrics = {
            'before': metrics_before,
            'after': metrics_after,
            'improvement': improvement
        }
        
        # 6. 저장
        if output_path is None:
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"{basename}_enhanced.wav"
        
        sf.write(output_path, y_normalized, sr)
        print(f"\n💾 향상된 음성 저장: {output_path}")
        
        # 7. 시각화 (metrics가 설정된 후!)
        if visualize:
            self._visualize_enhancement(
                y, y_normalized, sr, 
                metrics_before, metrics_after,
                audio_path
            )
        
        # 결과 출력
        print("\n" + "="*50)
        print("📈 음성 품질 개선 결과")
        print("="*50)
        print(f"  🔹 노이즈 감소: {improvement['noise_reduction']:.1f}%")
        print(f"  🔹 SNR 개선: {improvement['snr_improvement']:.1f}%")
        print(f"  🔹 다이나믹 레인지 최적화: {improvement['dynamic_range']:.1f}%")
        print(f"  🔹 전체 품질 향상: {improvement['overall']:.1f}%")
        print("="*50)
        
        return output_path, self.metrics
    
    def _analyze_noise_profile(self, y: np.ndarray, sr: int) -> Dict:
        """노이즈 프로파일 분석"""
        # 첫 0.5초를 노이즈 구간으로 가정 (상담 시작 전 침묵)
        noise_sample = y[:int(sr * 0.5)]
        
        # 노이즈 레벨 계산 (RMS)
        noise_level = 20 * np.log10(np.sqrt(np.mean(noise_sample**2)) + 1e-10)
        
        # 전체 신호 레벨
        signal_level = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
        
        # SNR 계산
        snr = signal_level - noise_level
        
        # 주파수 스펙트럼 분석
        noise_fft = np.abs(np.fft.rfft(noise_sample))
        noise_freq = np.fft.rfftfreq(len(noise_sample), 1/sr)
        
        return {
            'noise_level': noise_level,
            'signal_level': signal_level,
            'snr': snr,
            'noise_spectrum': noise_fft,
            'frequencies': noise_freq
        }
    
    def _spectral_gating(self, y: np.ndarray, sr: int, 
                        noise_profile: Dict) -> np.ndarray:
        """스펙트럴 게이팅을 통한 노이즈 제거"""
        # STFT 변환
        D = librosa.stft(y)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # 노이즈 게이트 임계값 (노이즈 레벨의 1.5배)
        threshold = np.percentile(magnitude, 20)
        
        # 게이팅 적용 (부드러운 전환)
        mask = magnitude > threshold
        magnitude_gated = magnitude * mask
        
        # 스무딩 (급격한 변화 방지)
        from scipy.ndimage import gaussian_filter
        mask_smooth = gaussian_filter(mask.astype(float), sigma=1.0)
        magnitude_gated = magnitude * mask_smooth
        
        # 역변환
        D_gated = magnitude_gated * np.exp(1j * phase)
        y_gated = librosa.istft(D_gated)
        
        return y_gated
    
    def _enhance_speech_band(self, y: np.ndarray, sr: int) -> np.ndarray:
        """음성 주파수 대역 강조 (300-3400Hz)"""
        # 대역통과 필터 설계
        nyquist = sr / 2
        low_freq = 300 / nyquist
        high_freq = min(3400 / nyquist, 0.99)  # Nyquist 한계 체크
        
        # Butterworth 필터
        sos = signal.butter(
            N=5,
            Wn=[low_freq, high_freq],
            btype='band',
            output='sos'
        )
        
        # 필터 적용
        y_filtered = signal.sosfilt(sos, y)
        
        # 원본과 믹싱 (과도한 필터링 방지)
        alpha = 0.7  # 필터링 강도
        y_enhanced = alpha * y_filtered + (1 - alpha) * y
        
        return y_enhanced
    
    def _normalize_dynamic_range(self, y: np.ndarray) -> np.ndarray:
        """다이나믹 레인지 정규화"""
        # RMS 정규화
        rms = np.sqrt(np.mean(y**2))
        target_rms = 0.1  # 목표 RMS 레벨
        
        if rms > 0:
            y_normalized = y * (target_rms / rms)
        else:
            y_normalized = y
        
        # 클리핑 방지
        max_val = np.max(np.abs(y_normalized))
        if max_val > 0.95:
            y_normalized = y_normalized * (0.95 / max_val)
        
        # 부드러운 압축 (컴프레서)
        threshold = 0.5
        ratio = 4.0  # 4:1 압축
        
        mask = np.abs(y_normalized) > threshold
        y_compressed = y_normalized.copy()
        y_compressed[mask] = np.sign(y_normalized[mask]) * (
            threshold + (np.abs(y_normalized[mask]) - threshold) / ratio
        )
        
        return y_compressed
    
    def _calculate_metrics(self, y: np.ndarray, sr: int, label: str) -> Dict:
        """음성 품질 지표 계산"""
        # RMS (Root Mean Square) - 평균 음량
        rms = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # 다이나믹 레인지
        max_val = np.max(np.abs(y))
        min_val = np.mean(np.abs(y[np.abs(y) > np.percentile(np.abs(y), 10)]))
        dynamic_range = 20 * np.log10((max_val / (min_val + 1e-10)) + 1e-10)
        
        # 스펙트럴 센트로이드 (음색 밝기)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)
        
        # 제로 크로싱 레이트 (노이즈 지표)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zcr)
        
        print(f"\n📊 {label} 음성 지표:")
        print(f"  • RMS 레벨: {rms_db:.2f} dB")
        print(f"  • 다이나믹 레인지: {dynamic_range:.2f} dB")
        print(f"  • 스펙트럴 센트로이드: {avg_centroid:.0f} Hz")
        print(f"  • 제로 크로싱 레이트: {avg_zcr:.4f}")
        
        return {
            'rms': rms,
            'rms_db': rms_db,
            'dynamic_range': dynamic_range,
            'spectral_centroid': avg_centroid,
            'zcr': avg_zcr
        }
    
    def _calculate_improvement(self, before: Dict, after: Dict) -> Dict:
        """개선율 계산"""
        # 노이즈 감소 (ZCR 감소율)
        noise_reduction = (1 - after['zcr'] / before['zcr']) * 100
        
        # SNR 개선 (RMS 증가율)
        snr_improvement = ((after['rms_db'] - before['rms_db']) / abs(before['rms_db'])) * 100
        
        # 다이나믹 레인지 최적화
        target_dynamic_range = 40  # 목표 다이나믹 레인지 (dB)
        before_diff = abs(before['dynamic_range'] - target_dynamic_range)
        after_diff = abs(after['dynamic_range'] - target_dynamic_range)
        dynamic_improvement = (1 - after_diff / before_diff) * 100 if before_diff > 0 else 0
        
        # 전체 품질 향상 (가중 평균)
        overall = (noise_reduction * 0.4 + abs(snr_improvement) * 0.3 + dynamic_improvement * 0.3)
        
        return {
            'noise_reduction': max(0, noise_reduction),
            'snr_improvement': snr_improvement,
            'dynamic_range': dynamic_improvement,
            'overall': overall
        }
    
    def _visualize_enhancement(self, y_before: np.ndarray, y_after: np.ndarray,
                              sr: int, metrics_before: Dict, metrics_after: Dict,
                              audio_path: str):
        """처리 전후 시각화"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('음성 품질 향상 결과 (Audio Enhancement Results)', fontsize=16, fontweight='bold')
        
        # 1. 파형 (Waveform)
        time_before = np.linspace(0, len(y_before)/sr, len(y_before))
        time_after = np.linspace(0, len(y_after)/sr, len(y_after))
        
        axes[0, 0].plot(time_before, y_before, alpha=0.7)
        axes[0, 0].set_title('원본 파형 (Original)', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_after, y_after, alpha=0.7, color='green')
        axes[0, 1].set_title('향상된 파형 (Enhanced)', fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2. 스펙트로그램 (Spectrogram)
        D_before = librosa.amplitude_to_db(np.abs(librosa.stft(y_before)), ref=np.max)
        D_after = librosa.amplitude_to_db(np.abs(librosa.stft(y_after)), ref=np.max)
        
        img1 = librosa.display.specshow(D_before, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, 0])
        axes[1, 0].set_title('원본 스펙트로그램', fontweight='bold')
        axes[1, 0].set_ylim(0, 4000)  # 음성 주파수 대역 포커스
        plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
        
        img2 = librosa.display.specshow(D_after, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, 1])
        axes[1, 1].set_title('향상된 스펙트로그램', fontweight='bold')
        axes[1, 1].set_ylim(0, 4000)
        plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
        
        # 3. 품질 지표 비교 (Metrics Comparison)
        metrics_labels = ['RMS (dB)', 'Dynamic\nRange (dB)', 'Spectral\nCentroid (Hz)', 'ZCR']
        before_values = [
            metrics_before['rms_db'],
            metrics_before['dynamic_range'],
            metrics_before['spectral_centroid']/100,  # 스케일 조정
            metrics_before['zcr']*100  # 스케일 조정
        ]
        after_values = [
            metrics_after['rms_db'],
            metrics_after['dynamic_range'],
            metrics_after['spectral_centroid']/100,
            metrics_after['zcr']*100
        ]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        axes[2, 0].bar(x - width/2, before_values, width, label='원본', alpha=0.8, color='blue')
        axes[2, 0].bar(x + width/2, after_values, width, label='향상', alpha=0.8, color='green')
        axes[2, 0].set_xlabel('지표 (Metrics)')
        axes[2, 0].set_ylabel('값 (Value)')
        axes[2, 0].set_title('품질 지표 비교', fontweight='bold')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(metrics_labels)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 4. 개선율 표시 (Improvement Rates)
        improvement = self.metrics['improvement']
        categories = ['노이즈 감소\n(Noise Reduction)', 
                     'SNR 개선\n(SNR Improvement)', 
                     '다이나믹 레인지\n(Dynamic Range)',
                     '전체 품질\n(Overall Quality)']
        improvements = [
            improvement['noise_reduction'],
            abs(improvement['snr_improvement']),
            improvement['dynamic_range'],
            improvement['overall']
        ]
        
        colors = ['red', 'orange', 'yellow', 'green']
        bars = axes[2, 1].bar(categories, improvements, color=colors, alpha=0.7)
        axes[2, 1].set_ylabel('개선율 (%)')
        axes[2, 1].set_title('품질 개선 지표', fontweight='bold')
        axes[2, 1].set_ylim(0, max(improvements) * 1.2)
        axes[2, 1].grid(True, alpha=0.3)
        
        # 막대 위에 수치 표시
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        output_filename = f"enhancement_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 시각화 저장: {output_filename}")
        plt.show()


# 간편 사용 함수
def enhance_audio(input_path: str, output_path: Optional[str] = None, 
                 visualize: bool = True) -> Tuple[str, Dict]:
    """
    음성 품질 향상 간편 실행 함수
    
    Args:
        input_path: 입력 음성 파일 경로
        output_path: 출력 파일 경로 (선택)
        visualize: 시각화 여부
        
    Returns:
        (향상된 음성 파일 경로, 개선 지표)
    """
    enhancer = AudioEnhancer(target_sr=16000)
    return enhancer.enhance(input_path, output_path, visualize)


if __name__ == "__main__":
    # 테스트 실행
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"🎵 음성 파일 처리: {input_file}")
        output_file, metrics = enhance_audio(input_file)
        print(f"\n✅ 처리 완료!")
        print(f"📁 출력 파일: {output_file}")
    else:
        print("사용법: python audio_enhancer.py [오디오파일경로]")
        print("예시: python audio_enhancer.py test.mp3")