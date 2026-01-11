"""
ComfyUI model_sampling 포팅

원본: ComfyUI/comfy/model_sampling.py, ComfyUI/comfy/ldm/modules/diffusionmodules/util.py
"""

import math
import torch


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """ComfyUI 원본 그대로"""
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def reshape_sigma(sigma, noise_dim):
    """ComfyUI 원본 그대로"""
    if sigma.nelement() == 1:
        return sigma.view(())
    else:
        return sigma.view(sigma.shape[:1] + (1,) * (noise_dim - 1))


class EPS:
    """
    ComfyUI EPS 예측 타입

    SDXL은 epsilon (노이즈) 예측 모델을 사용합니다.
    """

    def __init__(self, sigma_data=1.0):
        self.sigma_data = sigma_data

    def calculate_input(self, sigma, noise):
        """모델 입력 계산"""
        sigma = reshape_sigma(sigma, noise.ndim)
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        """denoised 이미지 계산"""
        sigma = reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """
        노이즈 스케일링

        Args:
            sigma: 현재 시그마 값
            noise: 노이즈 텐서
            latent_image: 원본 레이턴트 (img2img용, txt2img에서는 zero)
            max_denoise: 최대 디노이즈 여부 (denoise=1.0일 때 True)
        """
        sigma = reshape_sigma(sigma, noise.ndim)
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        """역 노이즈 스케일링 (EPS에서는 항등 변환)"""
        return latent


class V_PREDICTION(EPS):
    """V-prediction 타입 (일부 모델에서 사용)"""

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = reshape_sigma(sigma, model_output.ndim)
        return (model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
                - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5)


class ModelSamplingDiscrete(torch.nn.Module):
    """
    ComfyUI ModelSamplingDiscrete

    SDXL 기본값:
    - beta_schedule: "linear"
    - linear_start: 0.00085
    - linear_end: 0.012
    - timesteps: 1000
    """

    def __init__(self, model_config=None, zsnr=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.get("sampling_settings", {})
        else:
            sampling_settings = {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)
        timesteps = sampling_settings.get("timesteps", 1000)

        if zsnr is None:
            zsnr = sampling_settings.get("zsnr", False)

        self._register_schedule(
            given_betas=None,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=8e-3,
            zsnr=zsnr
        )
        self.sigma_data = 1.0

    def _register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, zsnr=False):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s
            )
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.zsnr = zsnr

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())
        # 캐시된 device-specific 텐서들
        self._cached_log_sigmas = None
        self._cached_device = None

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def _get_log_sigmas(self, device):
        """캐시된 log_sigmas 반환 (device별로 캐싱)"""
        if self._cached_log_sigmas is None or self._cached_device != device:
            self._cached_log_sigmas = self.log_sigmas.to(device)
            self._cached_device = device
        return self._cached_log_sigmas

    def timestep(self, sigma):
        """
        sigma → timestep 변환

        로그 공간에서 가장 가까운 timestep을 찾습니다.
        이것이 Diffusers와의 핵심 차이점 중 하나입니다.
        """
        log_sigma = sigma.log()
        # 캐시된 log_sigmas 사용
        log_sigmas = self._get_log_sigmas(sigma.device)
        dists = log_sigma - log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def sigma(self, timestep):
        """
        timestep → sigma 변환

        로그 공간에서 선형 보간합니다.
        이것이 Diffusers와의 핵심 차이점 중 하나입니다.
        """
        # 캐시된 log_sigmas 사용
        log_sigmas = self._get_log_sigmas(timestep.device)
        t = torch.clamp(timestep.float(), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        return log_sigma.exp()

    def percent_to_sigma(self, percent):
        """denoise 비율을 sigma로 변환"""
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()
