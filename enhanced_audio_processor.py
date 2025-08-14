# enhanced_audio_processor.py
"""
개선된 음성 품질 향상 모듈
- 적응형 노이즈 프로파일링
- 동적 파라미터 조정
- 메모리 효율 최적화
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class EnhancedAudioProcessor:
    """개선된 음성 품질 향상 클래스"""
    
    def __init__(self, target_sr: int = 16000):
        """
        Args:
            target_sr: 목표 샘플링 레이트 (ETRI STT 최적: 16000)
        """
        self.target_sr = target_sr
        self.metrics = {}
        
        # 최적화된 STFT 파라미터
        self.n_fft = 1024  # 줄임 (기존 2048)
        self.hop_length = 256  # 줄임 (기존 512)
        
    def process(self, 
                audio_path: str, 
                output_path: Optional[str] = None,
                aggressive: bool = False,
                auto_detect_noise: bool = True) -> Tuple[str, Dict]:
        """
        음성 품질 향상 메인 파이프라인 (간소화 + 최적화)
        
        Args:
            audio_path: 입력 음성 파일 경로
            output_path: 출력 파일 경로 (None이면 메모리 반환)
            aggressive: 공격적 노이즈 제거 모드
            auto_detect_noise: 자동 노이즈 구간 탐지
            
        Returns:
            (출력 경로 또는 오디오 배열, 개선 지표)
        """
        # 1. 오디오 로드
        y, sr = librosa.load(audio_path, sr=self.target_sr)
        original_rms = np.sqrt(np.mean(y**2))
        
        # 2. 노이즈 프로파일 분석 (개선)
        if auto_detect_noise:
            noise_profile = self._auto_detect_noise_profile(y, sr)
        else:
            noise_profile = self._analyze_noise_profile(y, sr)
        
        # 3. 적응형 스펙트럴 게이팅
        noise_level = noise_profile['noise_level']
        adaptive_threshold = self._calculate_adaptive_threshold(noise_level, aggressive)
        y_denoised = self._adaptive_spectral_gating(y, sr, noise_profile, adaptive_threshold)
        
        # 4. 음성 대역 최적화 (동적)
        speech_band = self._detect_speech_band(y_denoised, sr)
        y_enhanced = self._enhance_speech_band_dynamic(y_denoised, sr, speech_band)
        
        # 5. 스마트 정규화
        y_normalized = self._smart_normalize(y_enhanced, target_loudness=-20)
        
        # 6. 메트릭 계산
        self.metrics = self._calculate_improvement_metrics(y, y_normalized, sr)
        
        # 7. 저장 또는 반환
        if output_path:
            sf.write(output_path, y_normalized, sr)
            return output_path, self.metrics
        else:
            return y_normalized, self.metrics
    
    def _auto_detect_noise_profile(self, y: np.ndarray, sr: int) -> Dict:
        """자동으로 노이즈 구간을 찾아 프로파일 생성"""
        # 에너지 기반 무음 구간 탐지
        frame_length = int(0.025 * sr)  # 25ms 프레임
        hop_length = int(0.010 * sr)    # 10ms 홉
        
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.percentile(energy, 20)  # 하위 20%를 무음으로 간주
        
        # 무음 구간 추출
        silence_frames = energy < threshold
        
        # 연속된 무음 구간 찾기 (최소 0.3초)
        min_silence_frames = int(0.3 * sr / hop_length)
        noise_segments = []
        
        start = None
        count = 0
        for i, is_silence in enumerate(silence_frames):
            if is_silence:
                if start is None:
                    start = i
                count += 1
            else:
                if start is not None and count >= min_silence_frames:
                    noise_segments.append((start * hop_length, i * hop_length))
                start = None
                count = 0
        
        # 노이즈 샘플 수집
        if noise_segments:
            noise_samples = []
            for start, end in noise_segments[:3]:  # 최대 3개 구간 사용
                noise_samples.append(y[start:end])
            noise_sample = np.concatenate(noise_samples)
        else:
            # 무음 구간이 없으면 처음 0.5초 사용
            noise_sample = y[:int(sr * 0.5)]
        
        # 노이즈 특성 분석
        noise_level = 20 * np.log10(np.sqrt(np.mean(noise_sample**2)) + 1e-10)
        signal_level = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
        
        return {
            'noise_level': noise_level,
            'signal_level': signal_level,
            'snr': signal_level - noise_level,
            'noise_sample': noise_sample,
            'auto_detected': True
        }
    
    def _analyze_noise_profile(self, y: np.ndarray, sr: int) -> Dict:
        """기본 노이즈 프로파일 분석 (폴백)"""
        noise_sample = y[:int(sr * 0.5)]
        noise_level = 20 * np.log10(np.sqrt(np.mean(noise_sample**2)) + 1e-10)
        signal_level = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
        
        return {
            'noise_level': noise_level,
            'signal_level': signal_level,
            'snr': signal_level - noise_level,
            'noise_sample': noise_sample,
            'auto_detected': False
        }
    
    def _calculate_adaptive_threshold(self, noise_level: float, aggressive: bool) -> float:
        """노이즈 레벨에 따른 적응형 임계값 계산"""
        # 노이즈 레벨에 따라 임계값 조정
        if noise_level < -40:  # 조용한 환경
            base_threshold = 0.15
        elif noise_level < -30:  # 보통 환경
            base_threshold = 0.20
        else:  # 시끄러운 환경
            base_threshold = 0.25
        
        # 공격적 모드면 임계값 상향
        if aggressive:
            base_threshold *= 1.3
        
        return base_threshold
    
    def _adaptive_spectral_gating(self, y: np.ndarray, sr: int, 
                                 noise_profile: Dict, threshold_factor: float) -> np.ndarray:
        """적응형 스펙트럴 게이팅"""
        # STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # 동적 임계값 계산
        threshold = np.percentile(magnitude, threshold_factor * 100)
        
        # 주파수별 가중치 (음성 대역 보호)
        freq_bins = magnitude.shape[0]
        freq_weights = np.ones(freq_bins)
        
        # 300-3400Hz 구간 보호
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        freq_weights[speech_mask] = 0.7  # 음성 대역은 덜 제거
        
        # 가중치 적용된 마스크
        weighted_threshold = threshold * freq_weights[:, np.newaxis]
        mask = magnitude > weighted_threshold
        
        # 시간축 스무딩 (급격한 변화 방지)
        mask = gaussian_filter(mask.astype(float), sigma=[0.5, 1.5])
        
        # 적용
        magnitude_gated = magnitude * mask
        D_gated = magnitude_gated * np.exp(1j * phase)
        
        # 역변환
        y_gated = librosa.istft(D_gated, hop_length=self.hop_length)
        
        return y_gated
    
    def _detect_speech_band(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """음성 주파수 대역 자동 감지"""
        # 스펙트럴 센트로이드로 음성 중심 주파수 추정
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroids)
        
        # 성별/나이 추정 (센트로이드 기반)
        if mean_centroid < 1500:  # 저음 (남성/노인)
            return (200, 3000)
        elif mean_centroid < 2000:  # 중음 (일반)
            return (300, 3400)
        else:  # 고음 (여성/아동)
            return (350, 4000)
    
    def _enhance_speech_band_dynamic(self, y: np.ndarray, sr: int, 
                                    speech_band: Tuple[float, float]) -> np.ndarray:
        """동적 음성 대역 강조"""
        low_freq, high_freq = speech_band
        
        # Butterworth 필터 설계
        nyquist = sr / 2
        low_norm = low_freq / nyquist
        high_norm = min(high_freq / nyquist, 0.99)
        
        sos = signal.butter(N=4, Wn=[low_norm, high_norm], 
                           btype='band', output='sos')
        
        y_filtered = signal.sosfilt(sos, y)
        
        # 원본과 동적 믹싱 (SNR에 따라)
        filtered_rms = np.sqrt(np.mean(y_filtered**2))
        original_rms = np.sqrt(np.mean(y**2))
        
        if filtered_rms > 0 and original_rms > 0:
            snr_ratio = filtered_rms / original_rms
            mix_ratio = np.clip(0.5 + snr_ratio * 0.3, 0.6, 0.85)
        else:
            mix_ratio = 0.7
        
        y_enhanced = mix_ratio * y_filtered + (1 - mix_ratio) * y
        
        return y_enhanced
    
    def _smart_normalize(self, y: np.ndarray, target_loudness: float = -20) -> np.ndarray:
        """스마트 정규화 (LUFS 기반 시뮬레이션)"""
        # 간단한 LUFS 근사
        # K-weighting 필터 (간략화)
        b, a = signal.butter(2, [80, 3000], btype='band', fs=self.target_sr)
        y_weighted = signal.filtfilt(b, a, y)
        
        # 현재 라우드니스 추정
        current_loudness = 20 * np.log10(np.sqrt(np.mean(y_weighted**2)) + 1e-10)
        
        # 게인 계산
        gain_db = target_loudness - current_loudness
        gain_linear = 10 ** (gain_db / 20)
        
        # 리미터 적용 (클리핑 방지)
        max_gain = 0.95 / np.max(np.abs(y))
        final_gain = min(gain_linear, max_gain)
        
        y_normalized = y * final_gain
        
        # Soft clipping (부드러운 제한)
        threshold = 0.9
        mask = np.abs(y_normalized) > threshold
        if np.any(mask):
            over = np.abs(y_normalized[mask]) - threshold
            y_normalized[mask] = np.sign(y_normalized[mask]) * (threshold + np.tanh(over * 3) * 0.1)
        
        return y_normalized
    
    def _calculate_improvement_metrics(self, y_original: np.ndarray, 
                                      y_processed: np.ndarray, sr: int) -> Dict:
        """개선 지표 계산"""
        # RMS
        rms_orig = np.sqrt(np.mean(y_original**2))
        rms_proc = np.sqrt(np.mean(y_processed**2))
        
        # 스펙트럴 특성
        cent_orig = np.mean(librosa.feature.spectral_centroid(y=y_original, sr=sr)[0])
        cent_proc = np.mean(librosa.feature.spectral_centroid(y=y_processed, sr=sr)[0])
        
        # ZCR (노이즈 지표)
        zcr_orig = np.mean(librosa.feature.zero_crossing_rate(y_original)[0])
        zcr_proc = np.mean(librosa.feature.zero_crossing_rate(y_processed)[0])
        
        # 개선율 계산
        noise_reduction = max(0, (1 - zcr_proc / zcr_orig) * 100)
        clarity_improvement = (cent_proc / cent_orig - 1) * 100 if cent_orig > 0 else 0
        
        return {
            'before': {
                'rms_db': 20 * np.log10(rms_orig + 1e-10),
                'zcr': zcr_orig
            },
            'after': {
                'rms_db': 20 * np.log10(rms_proc + 1e-10),
                'zcr': zcr_proc
            },
            'improvement': {
                'noise_reduction': noise_reduction,
                'clarity': clarity_improvement,
                'overall': (noise_reduction + abs(clarity_improvement)) / 2
            }
        }


# 간편 사용 함수
def enhance_audio(input_path: str, output_path: Optional[str] = None, 
                 aggressive: bool = False) -> Tuple[any, Dict]:
    """
    음성 품질 향상 간편 실행
    
    Args:
        input_path: 입력 파일
        output_path: 출력 파일 (None이면 배열 반환)
        aggressive: 공격적 노이즈 제거
    
    Returns:
        (오디오 데이터 또는 경로, 메트릭)
    """
    processor = EnhancedAudioProcessor(target_sr=16000)
    return processor.process(input_path, output_path, aggressive)