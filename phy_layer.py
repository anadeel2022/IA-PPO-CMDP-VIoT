# phy_layer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Time and rate constants used across the codebase
SLOT_DURATION_S   = 1e-3          # 1 ms slot
CHANNEL_BW_HZ = 1_000_000          # 1 MHz per orthogonal resource channel
MAX_SPECTRAL_EFFICIENCY = 6.0        # capped Shannon mapping (bit/s/Hz)
CHANNEL_RATE_BPS = int(CHANNEL_BW_HZ * MAX_SPECTRAL_EFFICIENCY)
THROUGHPUT_SCALE = 1.0               # no artificial throughput inflation


@dataclass
class PHYCfg:
    sinr_mu_db: float = 12.0       # mean large-scale SINR (dB)
    sinr_sigma_db: float = 6.0     # log-normal spread (dB)
    rayleigh_scale: float = 1.0    # small-scale fading parameter
    shadow_sigma_db: float = 4.0   # log-normal shadowing std (dB)
    interf_db_mu: float = -6.0     # mean interference in dB relative to signal
    interf_db_sigma: float = 3.0   # spread of interference (dB)

class PHYLayer:
    """
    Lightweight PHY helper used by the HIoT environment.
    Provides:
      - slot_bits(sinr_db): capacity in bits for a SLOT_DURATION_S slot
      - energy_joules(tx_power_dbm, slot_s): TX energy consumption
      - sample_sinr_db(...): draw a SINR sample with optional fading/shadowing/interference
    """
    cfg = PHYCfg()

    @staticmethod
    def set_cfg(**kwargs) -> None:
        """Optionally adjust default PHY parameters at runtime."""
        for k, v in kwargs.items():
            if hasattr(PHYLayer.cfg, k):
                setattr(PHYLayer.cfg, k, v)

    # ---------- core mappings ----------
    @staticmethod
    def _db_to_linear(db: float) -> float:
        return float(10.0 ** (db / 10.0))

    @staticmethod
    def _linear_to_db(x: float) -> float:
        x = max(float(x), 1e-12)
        return float(10.0 * np.log10(x))

    # Shannon-like mapping with a conservative spectral efficiency cap
    @staticmethod
    def _spectral_efficiency_bps_per_hz(sinr_linear: float) -> float:
        # log2(1+SINR) with a modest cap to avoid unrealistic rates for very high SINR
        se = float(np.log2(1.0 + max(sinr_linear, 1e-12)))
        return float(min(se, MAX_SPECTRAL_EFFICIENCY))  # approximately a 64-QAM upper bound

    # ---------- capacity / energy ----------
    @staticmethod
    def slot_bits(sinr_db: float, bw_hz: float = CHANNEL_BW_HZ) -> int:
        """
        Compute the bits that can be sent in one slot at given SINR.
        By default, bandwidth is 1 MHz, consistent with CHANNEL_RATE_BPS.
        """
        sinr_lin = PHYLayer._db_to_linear(sinr_db)
        se = PHYLayer._spectral_efficiency_bps_per_hz(sinr_lin)
        bits = se * bw_hz * SLOT_DURATION_S * THROUGHPUT_SCALE
        # bound and sanitize
        if not np.isfinite(bits) or bits < 0.0:
            bits = 0.0
        return int(max(0, round(bits)))

    @staticmethod
    def energy_joules(tx_power_dbm: float, slot_s: float) -> float:
        """
        Convert TX power in dBm to Joules consumed across slot_s.
        P(dBm) -> P(mW) -> P(W) = P(mW)/1000; Energy = P(W) * time(s)
        """
        p_mw = 10.0 ** (tx_power_dbm / 10.0)
        p_w  = p_mw / 1000.0
        e_j  = p_w * float(slot_s)
        if not np.isfinite(e_j) or e_j < 0.0:
            e_j = 0.0
        return float(e_j)

    # ---------- SINR sampling ----------
    @staticmethod
    def sample_sinr_db(rng: np.random.Generator,
                       mu_db: float,
                       sigma_db: float,
                       enable_fading: bool = True,
                       enable_shadowing: bool = True,
                       enable_interference: bool = True) -> float:
        """
        Compose SINR(dB) from large-scale term (N(mu_db, sigma_db)),
        optional small-scale fading (Rayleigh), optional log-normal shadowing,
        and optional interference as a subtractive term in dB.
        """
        # Large-scale (path + avg interference baked into mu_db if desired)
        sinr_db = float(rng.normal(mu_db, sigma_db))

        # Shadowing (log-normal in dB)
        if enable_shadowing and PHYLayer.cfg.shadow_sigma_db > 0.0:
            sinr_db += float(rng.normal(0.0, PHYLayer.cfg.shadow_sigma_db))

        # Small-scale fading (Rayleigh amplitude -> power ~ Exp(1))
        if enable_fading:
            # Rayleigh(scale) amplitude => power = amp^2
            amp = rng.rayleigh(scale=max(PHYLayer.cfg.rayleigh_scale, 1e-6))
            power_lin = float(amp ** 2)
            sinr_db += PHYLayer._linear_to_db(power_lin)

        # Interference as a random backoff in dB
        if enable_interference:
            interf_db = float(rng.normal(PHYLayer.cfg.interf_db_mu, PHYLayer.cfg.interf_db_sigma))
            sinr_db += interf_db  # interf_db is typically negative

        # Sanitize extremely low/high values
        sinr_db = float(np.clip(sinr_db, -20.0, 40.0))
        return sinr_db
