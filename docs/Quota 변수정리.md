# Quota 변수정리

이 문서는 `yunjun` 브랜치의 Quota v1과 group-local Quota v2 변수 계약을 정리한다. 실험 점수나 최고 모델 기록은 포함하지 않는다.

## 공통 표기

- 각 버전은 그룹별 64개 변수이고, 세 그룹의 합집합은 76개다.
- 표의 `G1`, `G2`, `G3`는 각각 `kpx_group_1`, `kpx_group_2`, `kpx_group_3`를 뜻한다.
- `✓`는 해당 그룹 Quota에 포함됨을 뜻한다.
- `u`, `v`는 각각 동서 및 남북 바람 성분이다.
- `lead1`, `lead3`는 같은 issue에서 1시간 및 3시간 뒤 예보값이다.
- `roll3_mean`은 같은 issue 안에서 현재 시각을 포함한 최근 3개 예보시간 평균이다.
- 표의 순서는 중요도 순위가 아니다.

## v1 계산 범위

- 기본 `ldaps_*`: LDAPS 16개 격자의 `u/v`를 평균한 뒤 계산한다.
- 기본 `gfs_*`: 발전단지 최근접 GFS 5번 격자를 사용한다.
- `grid_max`, `grid_range`, `grid_p90`: LDAPS 16개 또는 GFS 9개 전체 격자에서 계산한다.
- `near`: 그룹 중심 최근접 격자를 사용한다. LDAPS는 G1=5번, G2=6번, G3=12번이고 GFS는 세 그룹 모두 5번이다.
- `met_ldaps_*`는 LDAPS 전체 격자 평균, `met_gfs_*`는 GFS 5번 격자다.

## Quota v1 전체 변수

| # | 변수 | G1 | G2 | G3 | 짧은 설명 |
|---:|---|:---:|:---:|:---:|---|
| 1 | `phys_ldaps_ws50max_grid_max_roll3_mean` | ✓ | ✓ | ✓ | LDAPS 50m-max 전 격자 최대풍속의 3시간 평균 |
| 2 | `phys_ldaps_ws50max_grid_max_lead1` | ✓ | ✓ | ✓ | LDAPS 50m-max 전 격자 최대풍속의 +1시간 예보 |
| 3 | `phys_ldaps_ws50max_grid_max` | ✓ | ✓ | ✓ | LDAPS 50m-max 16격자 중 최대풍속 |
| 4 | `phys_ldaps_ws50max_grid_max_lead3` | ✓ | ✓ | ✓ | LDAPS 50m-max 전 격자 최대풍속의 +3시간 예보 |
| 5 | `phys_ldaps_ws50max_grid_range` | ✓ | ✓ | ✓ | LDAPS 50m-max 격자 최대풍속−최소풍속 |
| 6 | `phys_ldaps_ws50max_east_west_gradient` | ✓ | ✓ | ✓ | LDAPS 50m-max 풍속의 동서 공간 기울기 |
| 7 | `phys_gfs_ws10_near_southwest` | ✓ | ✓ | ✓ | 최근접 GFS 10m 바람의 남서축 투영값 |
| 8 | `phys_gfs_ws850_east_west_gradient` | ✓ | ✓ | ✓ | GFS 850hPa 풍속의 동서 공간 기울기 |
| 9 | `phys_ldaps_ws50min_grid_range` | ✓ | ✓ | ✓ | LDAPS 50m-min 격자 최대풍속−최소풍속 |
| 10 | `phys_gfs_ws500_near_axis_cross` | ✓ | ✓ | ✓ | 최근접 GFS 500hPa 바람의 그룹 배치축 횡방향 성분 |
| 11 | `phys_ldaps_ws50min_grid_max` | ✓ | ✓ | ✓ | LDAPS 50m-min 16격자 중 최대풍속 |
| 12 | `phys_gfs_ws850_near_axis_cross` | ✓ | ✓ | ✓ | 최근접 GFS 850hPa 바람의 그룹 배치축 횡방향 성분 |
| 13 | `phys_gfs_ws500_east_west_gradient` | ✓ | ✓ | ✓ | GFS 500hPa 풍속의 동서 공간 기울기 |
| 14 | `phys_ldaps_ws10_grid_max` | ✓ |  | ✓ | LDAPS 10m 16격자 중 최대풍속 |
| 15 | `phys_gfs_ws100_upwind_minus_downwind` | ✓ |  | ✓ | GFS 100m 상류 가중풍속−하류 가중풍속 |
| 16 | `phys_gfs_ws850_upwind_minus_downwind` | ✓ |  | ✓ | GFS 850hPa 상류 가중풍속−하류 가중풍속 |
| 17 | `phys_gfs_ws850_grid_p90_roll3_mean` | ✓ | ✓ |  | GFS 850hPa 격자 풍속 90% 분위의 3시간 평균 |
| 18 | `phys_gfs_ws10_grid_max` | ✓ | ✓ |  | GFS 10m 9격자 중 최대풍속 |
| 19 | `gfs_ws500_speed` | ✓ | ✓ | ✓ | GFS 5번 격자 500hPa 풍속 |
| 20 | `ldaps_heightAboveGround_5_XBLWS` | ✓ | ✓ | ✓ | LDAPS 16격자 평균 5m 경계층 x바람 |
| 21 | `gfs_heightAboveGround_10_10u` | ✓ | ✓ | ✓ | GFS 5번 격자 10m u바람 |
| 22 | `ldaps_heightAboveGround_50_50MUmin` | ✓ |  | ✓ | LDAPS 평균 50m-min u바람 |
| 23 | `gfs_heightAboveGround_10_10v` | ✓ | ✓ | ✓ | GFS 5번 격자 10m v바람 |
| 24 | `ldaps_ws5_bl_speed` | ✓ | ✓ |  | LDAPS 평균 5m 경계층 벡터 풍속 |
| 25 | `ldaps_heightAboveGround_50_50MVmin` | ✓ | ✓ | ✓ | LDAPS 평균 50m-min v바람 |
| 26 | `gfs_heightAboveGround_100_100v` | ✓ | ✓ | ✓ | GFS 5번 격자 100m v바람 |
| 27 | `ldaps_heightAboveGround_50_50MVmax` | ✓ | ✓ | ✓ | LDAPS 평균 50m-max v바람 |
| 28 | `gfs_heightAboveGround_100_100u` | ✓ | ✓ |  | GFS 5번 격자 100m u바람 |
| 29 | `gfs_ws850_speed` | ✓ |  | ✓ | GFS 5번 격자 850hPa 풍속 |
| 30 | `gfs_ws10_speed_roll3_mean` | ✓ | ✓ | ✓ | GFS 5번 10m 풍속의 3시간 평균 |
| 31 | `gfs_ws850_speed_roll3_mean` | ✓ | ✓ | ✓ | GFS 5번 850hPa 풍속의 3시간 평균 |
| 32 | `gfs_ws850_speed_lead3` | ✓ | ✓ | ✓ | GFS 5번 850hPa 풍속의 +3시간 예보 |
| 33 | `gfs_ws850_speed_lead1` | ✓ | ✓ | ✓ | GFS 5번 850hPa 풍속의 +1시간 예보 |
| 34 | `gfs_ws10_speed_lead3` | ✓ | ✓ | ✓ | GFS 5번 10m 풍속의 +3시간 예보 |
| 35 | `ldaps_ws10_speed_lead3` | ✓ |  | ✓ | LDAPS 평균 10m 풍속의 +3시간 예보 |
| 36 | `gfs_surface_0_gust_lead3` | ✓ | ✓ | ✓ | GFS 5번 지상 돌풍의 +3시간 예보 |
| 37 | `gfs_surface_0_gust_roll3_mean` | ✓ |  |  | GFS 5번 지상 돌풍의 3시간 평균 |
| 38 | `ldaps_ws10_speed_roll3_mean` | ✓ |  | ✓ | LDAPS 평균 10m 풍속의 3시간 평균 |
| 39 | `gfs_ws100_speed_lead3` | ✓ | ✓ | ✓ | GFS 5번 100m 풍속의 +3시간 예보 |
| 40 | `phys_gfs_lapse_850_500` | ✓ | ✓ | ✓ | GFS 850hPa 온도−500hPa 온도 |
| 41 | `phys_ldaps_surface_pressure` | ✓ | ✓ | ✓ | LDAPS 전 격자 평균 지표기압 |
| 42 | `phys_gfs_surface_pressure` | ✓ | ✓ | ✓ | GFS 5번 격자 지표기압 |
| 43 | `phys_gfs_shortwave` | ✓ | ✓ | ✓ | GFS 5번 격자 하향 단파복사 |
| 44 | `phys_shear_gfs_100_10` | ✓ | ✓ | ✓ | GFS 100m/10m 풍속비의 로그 |
| 45 | `phys_gfs_air_density_x_gfs_ws850_speed_cube` | ✓ | ✓ |  | GFS 공기밀도 × 850hPa 풍속³ |
| 46 | `phys_shear_gfs_850_100` | ✓ | ✓ | ✓ | GFS 850hPa/100m 풍속비의 로그 |
| 47 | `phys_shear_ldaps_50max_10` | ✓ |  | ✓ | LDAPS 50m-max/10m 풍속비의 로그 |
| 48 | `phys_gfs_gust_factor` | ✓ | ✓ |  | GFS 지상 돌풍 ÷ 10m 풍속 |
| 49 | `cos_doy` | ✓ | ✓ | ✓ | 연중 날짜 cosine 주기값 |
| 50 | `sin_doy` | ✓ | ✓ | ✓ | 연중 날짜 sine 주기값 |
| 51 | `hdw_sin` | ✓ | ✓ | ✓ | 요일 sine + 시간 sine |
| 52 | `hour_day_year_sin` | ✓ | ✓ | ✓ | 연중 날짜 sine + 시간 sine |
| 53 | `hour_day_year_cos` | ✓ | ✓ | ✓ | 연중 날짜 cosine + 시간 cosine |
| 54 | `sin_hod` | ✓ | ✓ | ✓ | 하루 중 시간의 sine 주기값 |
| 55 | `cos_hod` | ✓ | ✓ | ✓ | 하루 중 시간의 cosine 주기값 |
| 56 | `met_gfs_isobaricInhPa_700_t` | ✓ | ✓ | ✓ | GFS 5번 격자 700hPa 기온 |
| 57 | `met_gfs_isobaricInhPa_850_r` | ✓ | ✓ | ✓ | GFS 5번 격자 850hPa 상대습도 |
| 58 | `met_ldaps_surface_0_sp` | ✓ | ✓ | ✓ | LDAPS 전 격자 평균 지표기압 |
| 59 | `met_gfs_surface_0_dswrf` | ✓ | ✓ | ✓ | GFS 5번 격자 하향 단파복사 |
| 60 | `gfs_isobaricInhPa_850_u` | ✓ | ✓ | ✓ | GFS 5번 격자 850hPa u바람 |
| 61 | `gfs_isobaricInhPa_850_v` | ✓ | ✓ | ✓ | GFS 5번 격자 850hPa v바람 |
| 62 | `gfs_heightAboveGround_80_u` | ✓ | ✓ | ✓ | GFS 5번 격자 80m u바람 |
| 63 | `gfs_heightAboveGround_80_v` | ✓ | ✓ | ✓ | GFS 5번 격자 80m v바람 |
| 64 | `gfs_ws100_speed` | ✓ | ✓ | ✓ | GFS 5번 격자 100m 풍속 |
| 65 | `phys_gfs_ws850_grid_max_roll3_mean` |  | ✓ | ✓ | GFS 850hPa 9격자 최대풍속의 3시간 평균 |
| 66 | `phys_gfs_ws100_grid_max` |  | ✓ |  | GFS 100m 9격자 중 최대풍속 |
| 67 | `phys_gfs_ws850_grid_max` |  | ✓ |  | GFS 850hPa 9격자 중 최대풍속 |
| 68 | `gfs_ws10_speed` |  | ✓ |  | GFS 5번 격자 10m 풍속 |
| 69 | `ldaps_heightAboveGround_10_10v` |  | ✓ | ✓ | LDAPS 16격자 평균 10m v바람 |
| 70 | `gfs_ws100_speed_roll3_mean` |  | ✓ |  | GFS 5번 100m 풍속의 3시간 평균 |
| 71 | `gfs_ws10_speed_lead1` |  | ✓ |  | GFS 5번 10m 풍속의 +1시간 예보 |
| 72 | `ldaps_ws50_max_speed_lead3` |  | ✓ | ✓ | LDAPS 평균 50m-max 풍속의 +3시간 예보 |
| 73 | `phys_ldaps_shortwave` |  | ✓ | ✓ | LDAPS 전 격자 평균 하향 단파복사 |
| 74 | `phys_gfs_ws850_grid_max_lead3` |  |  | ✓ | GFS 850hPa 9격자 최대풍속의 +3시간 예보 |
| 75 | `ldaps_heightAboveGround_5_YBLWS` |  |  | ✓ | LDAPS 16격자 평균 5m 경계층 y바람 |
| 76 | `phys_gfs_gust_minus_ws10` |  |  | ✓ | GFS 지상 돌풍−10m 풍속 |

## v2 계산 범위

Quota v2는 변수 공식을 바꾸지 않고 계산에 사용하는 NWP 공간 범위를 그룹별 고정 계약으로 바꾼다. 격자와 가중치는 연도, outer fold 및 train/validation 구분에 따라 다시 선택하지 않는다.

| 그룹 | LDAPS 고정 격자와 가중 개수 | GFS 고정 격자와 가중 개수 |
|---|---|---|
| G1 | 2번×3, 12번×3 | 2번×2, 4번×3, 7번×1 |
| G2 | 2번×2, 3번×1, 7번×2, 12번×1 | 2번×1, 4번×5 |
| G3 | 3번×1, 12번×1, 13번×3 | 2번×4, 4번×1 |

1. 위 표의 격자 구성과 가중 개수를 모든 연도와 outer fold에 동일하게 적용한다.
2. 각 고정 격자의 전체 NWP 열을 가중 개수만큼 모아 그룹 패널을 만든다.
3. 패널 가중 평균을 그룹 중심 좌표의 `group_anchor`로 추가한다.
4. 기본값, 공간 통계, 축 투영, 물리 파생량 및 시간 파생량을 이 패널에서 다시 계산한다.
5. 시간에 따라 NWP 수치만 달라지며, 각 변수의 격자 정의와 계산식은 달라지지 않는다.

v2 변수명은 기존 이름 앞에 `gqv2__`를 붙인다. v1과 그룹별 변수 선택 개수는 동일하므로 모델 입력 차원도 동일하다.

## Quota v2 전체 변수

| # | 변수 | G1 | G2 | G3 | 짧은 설명 |
|---:|---|:---:|:---:|:---:|---|
| 1 | `gqv2__phys_ldaps_ws50max_grid_max_roll3_mean` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-max 최대풍속의 3시간 평균 |
| 2 | `gqv2__phys_ldaps_ws50max_grid_max_lead1` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-max 최대풍속의 +1시간 예보 |
| 3 | `gqv2__phys_ldaps_ws50max_grid_max` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-max 최대풍속 |
| 4 | `gqv2__phys_ldaps_ws50max_grid_max_lead3` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-max 최대풍속의 +3시간 예보 |
| 5 | `gqv2__phys_ldaps_ws50max_grid_range` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-max 풍속 범위 |
| 6 | `gqv2__phys_ldaps_ws50max_east_west_gradient` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-max 동서 기울기 |
| 7 | `gqv2__phys_gfs_ws10_near_southwest` | ✓ | ✓ | ✓ | 그룹 GFS anchor 10m 바람의 남서축 투영값 |
| 8 | `gqv2__phys_gfs_ws850_east_west_gradient` | ✓ | ✓ | ✓ | 그룹 GFS 패널 850hPa 동서 기울기 |
| 9 | `gqv2__phys_ldaps_ws50min_grid_range` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-min 풍속 범위 |
| 10 | `gqv2__phys_gfs_ws500_near_axis_cross` | ✓ | ✓ | ✓ | 그룹 GFS anchor 500hPa 횡방향 성분 |
| 11 | `gqv2__phys_ldaps_ws50min_grid_max` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 50m-min 최대풍속 |
| 12 | `gqv2__phys_gfs_ws850_near_axis_cross` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa 횡방향 성분 |
| 13 | `gqv2__phys_gfs_ws500_east_west_gradient` | ✓ | ✓ | ✓ | 그룹 GFS 패널 500hPa 동서 기울기 |
| 14 | `gqv2__phys_ldaps_ws10_grid_max` | ✓ |  | ✓ | 그룹 LDAPS 패널 10m 최대풍속 |
| 15 | `gqv2__phys_gfs_ws100_upwind_minus_downwind` | ✓ |  | ✓ | 그룹 GFS 패널 100m 상류−하류 가중풍속 |
| 16 | `gqv2__phys_gfs_ws850_upwind_minus_downwind` | ✓ |  | ✓ | 그룹 GFS 패널 850hPa 상류−하류 가중풍속 |
| 17 | `gqv2__phys_gfs_ws850_grid_p90_roll3_mean` | ✓ | ✓ |  | 그룹 GFS 패널 850hPa p90의 3시간 평균 |
| 18 | `gqv2__phys_gfs_ws10_grid_max` | ✓ | ✓ |  | 그룹 GFS 패널 10m 최대풍속 |
| 19 | `gqv2__gfs_ws500_speed` | ✓ | ✓ | ✓ | 그룹 GFS anchor 500hPa 풍속 |
| 20 | `gqv2__ldaps_heightAboveGround_5_XBLWS` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 평균 5m 경계층 x바람 |
| 21 | `gqv2__gfs_heightAboveGround_10_10u` | ✓ | ✓ | ✓ | 그룹 GFS anchor 10m u바람 |
| 22 | `gqv2__ldaps_heightAboveGround_50_50MUmin` | ✓ |  | ✓ | 그룹 LDAPS 패널 평균 50m-min u바람 |
| 23 | `gqv2__gfs_heightAboveGround_10_10v` | ✓ | ✓ | ✓ | 그룹 GFS anchor 10m v바람 |
| 24 | `gqv2__ldaps_ws5_bl_speed` | ✓ | ✓ |  | 그룹 LDAPS 패널 평균 5m 경계층 풍속 |
| 25 | `gqv2__ldaps_heightAboveGround_50_50MVmin` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 평균 50m-min v바람 |
| 26 | `gqv2__gfs_heightAboveGround_100_100v` | ✓ | ✓ | ✓ | 그룹 GFS anchor 100m v바람 |
| 27 | `gqv2__ldaps_heightAboveGround_50_50MVmax` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 평균 50m-max v바람 |
| 28 | `gqv2__gfs_heightAboveGround_100_100u` | ✓ | ✓ |  | 그룹 GFS anchor 100m u바람 |
| 29 | `gqv2__gfs_ws850_speed` | ✓ |  | ✓ | 그룹 GFS anchor 850hPa 풍속 |
| 30 | `gqv2__gfs_ws10_speed_roll3_mean` | ✓ | ✓ | ✓ | 그룹 GFS anchor 10m 풍속의 3시간 평균 |
| 31 | `gqv2__gfs_ws850_speed_roll3_mean` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa 풍속의 3시간 평균 |
| 32 | `gqv2__gfs_ws850_speed_lead3` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa 풍속의 +3시간 예보 |
| 33 | `gqv2__gfs_ws850_speed_lead1` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa 풍속의 +1시간 예보 |
| 34 | `gqv2__gfs_ws10_speed_lead3` | ✓ | ✓ | ✓ | 그룹 GFS anchor 10m 풍속의 +3시간 예보 |
| 35 | `gqv2__ldaps_ws10_speed_lead3` | ✓ |  | ✓ | 그룹 LDAPS 패널 평균 10m 풍속의 +3시간 예보 |
| 36 | `gqv2__gfs_surface_0_gust_lead3` | ✓ | ✓ | ✓ | 그룹 GFS anchor 돌풍의 +3시간 예보 |
| 37 | `gqv2__gfs_surface_0_gust_roll3_mean` | ✓ |  |  | 그룹 GFS anchor 돌풍의 3시간 평균 |
| 38 | `gqv2__ldaps_ws10_speed_roll3_mean` | ✓ |  | ✓ | 그룹 LDAPS 패널 평균 10m 풍속의 3시간 평균 |
| 39 | `gqv2__gfs_ws100_speed_lead3` | ✓ | ✓ | ✓ | 그룹 GFS anchor 100m 풍속의 +3시간 예보 |
| 40 | `gqv2__phys_gfs_lapse_850_500` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa 온도−500hPa 온도 |
| 41 | `gqv2__phys_ldaps_surface_pressure` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 평균 지표기압 |
| 42 | `gqv2__phys_gfs_surface_pressure` | ✓ | ✓ | ✓ | 그룹 GFS anchor 지표기압 |
| 43 | `gqv2__phys_gfs_shortwave` | ✓ | ✓ | ✓ | 그룹 GFS anchor 하향 단파복사 |
| 44 | `gqv2__phys_shear_gfs_100_10` | ✓ | ✓ | ✓ | 그룹 GFS anchor 100m/10m 풍속비 로그 |
| 45 | `gqv2__phys_gfs_air_density_x_gfs_ws850_speed_cube` | ✓ | ✓ |  | 그룹 GFS 공기밀도 × 850hPa 풍속³ |
| 46 | `gqv2__phys_shear_gfs_850_100` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa/100m 풍속비 로그 |
| 47 | `gqv2__phys_shear_ldaps_50max_10` | ✓ |  | ✓ | 그룹 LDAPS 50m-max/10m 풍속비 로그 |
| 48 | `gqv2__phys_gfs_gust_factor` | ✓ | ✓ |  | 그룹 GFS anchor 돌풍 ÷ 10m 풍속 |
| 49 | `gqv2__cos_doy` | ✓ | ✓ | ✓ | 연중 날짜 cosine 주기값 |
| 50 | `gqv2__sin_doy` | ✓ | ✓ | ✓ | 연중 날짜 sine 주기값 |
| 51 | `gqv2__hdw_sin` | ✓ | ✓ | ✓ | 요일 sine + 시간 sine |
| 52 | `gqv2__hour_day_year_sin` | ✓ | ✓ | ✓ | 연중 날짜 sine + 시간 sine |
| 53 | `gqv2__hour_day_year_cos` | ✓ | ✓ | ✓ | 연중 날짜 cosine + 시간 cosine |
| 54 | `gqv2__sin_hod` | ✓ | ✓ | ✓ | 하루 중 시간의 sine 주기값 |
| 55 | `gqv2__cos_hod` | ✓ | ✓ | ✓ | 하루 중 시간의 cosine 주기값 |
| 56 | `gqv2__met_gfs_isobaricInhPa_700_t` | ✓ | ✓ | ✓ | 그룹 GFS anchor 700hPa 기온 |
| 57 | `gqv2__met_gfs_isobaricInhPa_850_r` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa 상대습도 |
| 58 | `gqv2__met_ldaps_surface_0_sp` | ✓ | ✓ | ✓ | 그룹 LDAPS 패널 평균 지표기압 |
| 59 | `gqv2__met_gfs_surface_0_dswrf` | ✓ | ✓ | ✓ | 그룹 GFS anchor 하향 단파복사 |
| 60 | `gqv2__gfs_isobaricInhPa_850_u` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa u바람 |
| 61 | `gqv2__gfs_isobaricInhPa_850_v` | ✓ | ✓ | ✓ | 그룹 GFS anchor 850hPa v바람 |
| 62 | `gqv2__gfs_heightAboveGround_80_u` | ✓ | ✓ | ✓ | 그룹 GFS anchor 80m u바람 |
| 63 | `gqv2__gfs_heightAboveGround_80_v` | ✓ | ✓ | ✓ | 그룹 GFS anchor 80m v바람 |
| 64 | `gqv2__gfs_ws100_speed` | ✓ | ✓ | ✓ | 그룹 GFS anchor 100m 풍속 |
| 65 | `gqv2__phys_gfs_ws850_grid_max_roll3_mean` |  | ✓ | ✓ | 그룹 GFS 패널 850hPa 최대풍속의 3시간 평균 |
| 66 | `gqv2__phys_gfs_ws100_grid_max` |  | ✓ |  | 그룹 GFS 패널 100m 최대풍속 |
| 67 | `gqv2__phys_gfs_ws850_grid_max` |  | ✓ |  | 그룹 GFS 패널 850hPa 최대풍속 |
| 68 | `gqv2__gfs_ws10_speed` |  | ✓ |  | 그룹 GFS anchor 10m 풍속 |
| 69 | `gqv2__ldaps_heightAboveGround_10_10v` |  | ✓ | ✓ | 그룹 LDAPS 패널 평균 10m v바람 |
| 70 | `gqv2__gfs_ws100_speed_roll3_mean` |  | ✓ |  | 그룹 GFS anchor 100m 풍속의 3시간 평균 |
| 71 | `gqv2__gfs_ws10_speed_lead1` |  | ✓ |  | 그룹 GFS anchor 10m 풍속의 +1시간 예보 |
| 72 | `gqv2__ldaps_ws50_max_speed_lead3` |  | ✓ | ✓ | 그룹 LDAPS 패널 평균 50m-max 풍속의 +3시간 예보 |
| 73 | `gqv2__phys_ldaps_shortwave` |  | ✓ | ✓ | 그룹 LDAPS 패널 평균 하향 단파복사 |
| 74 | `gqv2__phys_gfs_ws850_grid_max_lead3` |  |  | ✓ | 그룹 GFS 패널 850hPa 최대풍속의 +3시간 예보 |
| 75 | `gqv2__ldaps_heightAboveGround_5_YBLWS` |  |  | ✓ | 그룹 LDAPS 패널 평균 5m 경계층 y바람 |
| 76 | `gqv2__phys_gfs_gust_minus_ws10` |  |  | ✓ | 그룹 GFS anchor 돌풍−10m 풍속 |

## TCN에서의 사용

- v1 control: 그룹별 Quota v1 64개 + 터빈별 고정 격자 보조입력을 사용한다.
- v2: Quota v1 64개만 같은 개수의 고정 격자 group-local Quota v2로 교체한다.
- 두 버전 모두 터빈별 `wake_exposure`, `wake_upstream_count`와 아래 고정 격자 변수 4개를 사용한다.
  - `fixedgrid_ws_raw`: 고정 LDAPS 격자의 원시 풍속
  - `fixedgrid_ws_cube`: 고정 격자 원시 풍속의 세제곱
  - `fixedgrid_wd_sin`, `fixedgrid_wd_cos`: 고정 격자 풍향 성분
- 터빈별 격자와 높이는 위 v2 LDAPS 배치와 동일하다. `50m midpoint`는 50m max/min의 `u`, `v` 성분을 각각 평균한 뒤 풍속과 풍향을 계산한다.
- SCADA 풍속으로 fold마다 격자를 재선택하거나 `slope/intercept`를 적합하지 않는다.
- 두 버전 모두 G1/G2는 총 100개, G3는 총 94개 입력을 사용한다.
- 모델 구조, FiCR-only loss, 터빈별 local panel 및 그룹 target 직접 예측 구조는 바꾸지 않는다.
