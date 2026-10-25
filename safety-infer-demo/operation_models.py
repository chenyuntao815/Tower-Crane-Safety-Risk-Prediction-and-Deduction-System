"""Validated manual-state contracts for short-horizon operation comparison."""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CraneState(Contract):
    slew_deg: float = Field(35, ge=-360, le=360)
    slew_rate_deg_s: float = Field(3, ge=-8, le=8)
    radius_m: float = Field(35, ge=2, le=80)
    load_t: float = Field(5, gt=0, le=30)
    rope_length_m: float = Field(18, ge=2, le=100)
    wind_m_s: float = Field(8, ge=0, le=40)
    wind_direction_deg: float = Field(90, ge=-360, le=360)
    # Fixed world X / Z axes, not rotating radial / tangential axes.
    swing_x_deg: float = Field(1, ge=-12, le=12)
    swing_z_deg: float = Field(0, ge=-12, le=12)
    swing_rate_x_deg_s: float = Field(0, ge=-5, le=5)
    swing_rate_z_deg_s: float = Field(0, ge=-5, le=5)
    mast_x_mm: float | None = Field(None, ge=-1000, le=1000)
    mast_z_mm: float | None = Field(None, ge=-1000, le=1000)
    mast_rate_x_mm_s: float = Field(0, ge=-200, le=200)
    mast_rate_z_mm_s: float = Field(0, ge=-200, le=200)


class Equipment(Contract):
    tower_height_m: float = Field(48, ge=12, le=120)
    jib_length_m: float = Field(55, ge=15, le=80)
    attachment_heights_m: list[float] = Field(default_factory=lambda: [18, 36], max_length=8)
    mast_ei_kn_m2: float = Field(2_500_000, ge=100_000, le=100_000_000)
    mast_frequency_hz: float = Field(.65, ge=.15, le=2)
    mast_damping: float = Field(.05, ge=.01, le=.3)
    swing_damping: float = Field(.025, ge=.005, le=.3)
    tower_wind_area_m2: float = Field(85, ge=1, le=500)
    payload_wind_area_m2: float = Field(8, ge=.1, le=80)
    drag_coefficient: float = Field(1.3, ge=.3, le=3)
    attachment_lever_arm_m: float = Field(5, ge=1, le=15)
    rated_load_t: float = Field(12, gt=0, le=50)
    rated_lift_moment_kn_m: float = Field(2400, gt=0, le=20000)
    max_free_height_m: float = Field(25, ge=1, le=120)
    payload_radius_m: float = Field(.6, ge=.1, le=5)


class Uncertainty(Contract):
    gust_std_m_s: float = Field(1.5, ge=0, le=8)
    gust_correlation_s: float = Field(1.5, ge=.2, le=10)
    wind_bias_m_s: float = Field(.4, ge=0, le=4)
    wind_direction_error_deg: float = Field(10, ge=0, le=45)
    swing_error_deg: float = Field(.3, ge=0, le=2)
    swing_rate_error_deg_s: float = Field(.15, ge=0, le=1)
    slew_error_deg: float = Field(.2, ge=0, le=2)
    slew_rate_error_deg_s: float = Field(.08, ge=0, le=.5)
    relative_parameter_error: float = Field(.1, ge=0, le=.35)
    mast_prior_std_mm: float = Field(8, ge=.1, le=100)
    mast_measurement_std_mm: float = Field(2, ge=.1, le=100)


class Limits(Contract):
    swing_deg: float = Field(6, gt=0, le=12)
    mast_moment_kn_m: float = Field(3000, gt=0, le=20000)
    attachment_axial_kn: float = Field(260, gt=0, le=5000)
    min_clearance_m: float = Field(2, ge=0, le=20)
    swing_source: str = Field("演示阈值；请填写吊物控制要求或专项方案", min_length=1, max_length=300)
    mast_source: str = Field("演示阈值；请填写设备验算文件及工况", min_length=1, max_length=300)
    attachment_source: str = Field("演示阈值；请填写附着设计允许轴力", min_length=1, max_length=300)
    clearance_source: str = Field("演示阈值；请填写现场障碍物与安全距离要求", min_length=1, max_length=300)


class Obstacle(Contract):
    enabled: bool = False
    x_m: float = Field(22, ge=-150, le=150)
    z_m: float = Field(30, ge=-150, le=150)
    radius_m: float = Field(3, ge=.1, le=30)


class Controls(Contract):
    early_deceleration_s: float = Field(6, ge=1, le=15)
    reduced_speed_fraction: float = Field(.4, ge=.05, le=.9)
    smooth_braking_s: float = Field(6, ge=1, le=15)
    immediate_braking_s: float = Field(.8, ge=.2, le=3)
    max_acceleration_deg_s2: float = Field(1.5, gt=0, le=15)
    max_jerk_deg_s3: float = Field(4, gt=0, le=50)
    emergency_acceleration_deg_s2: float = Field(10, gt=0, le=50)
    emergency_jerk_deg_s3: float = Field(100, gt=0, le=1000)
    emergency_enabled: bool = True
    remaining_slew_deg: float = Field(30, ge=-180, le=180)
    restart_delay_s: float = Field(10, ge=0, le=600)
    time_cost_per_s: float = Field(1, ge=0, le=1000)
    stop_cost: float = Field(20, ge=0, le=100000)
    risk_cost: float = Field(1000, gt=0, le=10000000)
    risk_budget: float = Field(.1, ge=0, le=1)


class OperationRequest(Contract):
    state: CraneState = Field(default_factory=CraneState)
    equipment: Equipment = Field(default_factory=Equipment)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    limits: Limits = Field(default_factory=Limits)
    obstacle: Obstacle = Field(default_factory=Obstacle)
    controls: Controls = Field(default_factory=Controls)
    horizon_s: int = Field(12, ge=4, le=20)
    samples: int = Field(256, ge=64, le=1024)
    seed: int = Field(20261001, ge=0, le=4294967295)

    @model_validator(mode="after")
    def geometry_and_task(self):
        s, e = self.state, self.equipment
        if s.radius_m > e.jib_length_m:
            raise ValueError("幅度不能大于臂长")
        if s.rope_length_m + e.payload_radius_m >= e.tower_height_m:
            raise ValueError("绳长与吊物半径之和必须小于塔高")
        a = e.attachment_heights_m
        if a != sorted(set(a)) or any(v <= 0 or v >= e.tower_height_m for v in a):
            raise ValueError("附着高度须升序、不重复、大于零且低于塔高")
        if self.controls.remaining_slew_deg * s.slew_rate_deg_s < 0:
            raise ValueError("剩余回转角必须与当前回转方向一致")
        if abs(s.slew_rate_deg_s) < .01 and abs(self.controls.remaining_slew_deg) > .01:
            raise ValueError("当前角速度为零时请将剩余回转角设为零；本轮比较不包含启动操作")
        return self


class ValidationRequest(Contract):
    rows: list[dict[str, float | None]] = Field(min_length=1, max_length=10000)
