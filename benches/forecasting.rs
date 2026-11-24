use cartoboost_core::forecasting::{
    ArimaForecaster, AutoARIMAForecaster, ForecastFrame, ForecastFrequency, ForecastRow,
    ForecastWindow, Forecaster, RollingOriginSplitter,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn ts(hour: i64) -> chrono::NaiveDateTime {
    chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
        .expect("valid benchmark date")
        .and_hms_opt(0, 0, 0)
        .expect("valid benchmark timestamp")
        + chrono::Duration::hours(hour)
}

fn taxi_lane_frame(series_count: usize, rows_per_series: usize) -> ForecastFrame {
    let mut rows = Vec::with_capacity(series_count * rows_per_series);
    for series in 0..series_count {
        let pu = series % 263;
        let do_zone = (series * 17 + 3) % 263;
        let lane_bias = (pu as f64 * 0.07 + do_zone as f64 * 0.03).sin() * 8.0;
        for hour in 0..rows_per_series {
            let daily = ((hour % 24) as f64 / 24.0 * std::f64::consts::TAU).sin() * 12.0;
            let weekly = ((hour % 168) as f64 / 168.0 * std::f64::consts::TAU).cos() * 5.0;
            let trend = hour as f64 * 0.015;
            let target = 80.0 + lane_bias + daily + weekly + trend;
            rows.push(ForecastRow::new(
                format!("PU{pu}->DO{do_zone}"),
                ts(hour as i64),
                target,
            ));
        }
    }
    ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("valid benchmark frame")
}

fn bench_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("forecasting");
    group.sample_size(10);

    for &(series_count, rows_per_series, horizon) in &[(8, 240, 24), (64, 336, 24)] {
        let frame = taxi_lane_frame(series_count, rows_per_series);
        group.bench_with_input(
            BenchmarkId::new(
                "arima_fit_predict",
                format!("{series_count}lanes_{rows_per_series}hours_h{horizon}"),
            ),
            &frame,
            |bench, frame| {
                bench.iter(|| {
                    let mut model = ArimaForecaster::new(2, 1, 1).expect("valid arima order");
                    model.fit(black_box(frame)).expect("benchmark arima fits");
                    black_box(
                        model
                            .predict(black_box(horizon))
                            .expect("benchmark predicts"),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "auto_arima_fit_predict",
                format!("{series_count}lanes_{rows_per_series}hours_h{horizon}"),
            ),
            &frame,
            |bench, frame| {
                bench.iter(|| {
                    let mut model =
                        AutoARIMAForecaster::with_max_order(3, 1, 2).expect("valid grid");
                    model
                        .fit(black_box(frame))
                        .expect("benchmark auto_arima fits");
                    black_box(
                        model
                            .predict(black_box(horizon))
                            .expect("benchmark predicts"),
                    )
                });
            },
        );
    }

    let frame = taxi_lane_frame(8, 2400);
    let splitter = RollingOriginSplitter::new(24, 1, 168, None, Some(3), ForecastWindow::Expanding)
        .expect("valid rolling-origin splitter");
    group.bench_function("rolling_origin_last_3_8lanes_2400hours", |bench| {
        bench.iter(|| black_box(splitter.split(black_box(&frame)).expect("benchmark folds")));
    });

    group.finish();
}

criterion_group!(forecasting_benches, bench_forecasting);
criterion_main!(forecasting_benches);
