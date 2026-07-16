fn calendar_feature_name(feature: &CalendarFeature) -> String {
    match feature {
        CalendarFeature::DayOfWeek => "calendar_day_of_week".to_string(),
        CalendarFeature::DayOfWeekSin => "calendar_day_of_week_sin".to_string(),
        CalendarFeature::DayOfWeekCos => "calendar_day_of_week_cos".to_string(),
        CalendarFeature::Month => "calendar_month".to_string(),
        CalendarFeature::MonthSin => "calendar_month_sin".to_string(),
        CalendarFeature::MonthCos => "calendar_month_cos".to_string(),
        CalendarFeature::Day => "calendar_day".to_string(),
        CalendarFeature::DaySin => "calendar_day_sin".to_string(),
        CalendarFeature::DayCos => "calendar_day_cos".to_string(),
        CalendarFeature::MonthStart => "calendar_month_start".to_string(),
        CalendarFeature::MonthMiddle => "calendar_month_middle".to_string(),
        CalendarFeature::MonthEnd => "calendar_month_end".to_string(),
        CalendarFeature::DayOfYear => "calendar_day_of_year".to_string(),
        CalendarFeature::ElapsedIndex => "calendar_elapsed_index".to_string(),
        CalendarFeature::ElapsedPhase(_) => "calendar_elapsed_phase".to_string(),
    }
}

fn calendar_feature_allows_covariate_interaction(feature: &CalendarFeature) -> bool {
    match feature {
        CalendarFeature::DayOfWeek
        | CalendarFeature::DayOfWeekSin
        | CalendarFeature::DayOfWeekCos
        | CalendarFeature::Month
        | CalendarFeature::MonthSin
        | CalendarFeature::MonthCos
        | CalendarFeature::Day
        | CalendarFeature::DaySin
        | CalendarFeature::DayCos
        | CalendarFeature::MonthStart
        | CalendarFeature::MonthMiddle
        | CalendarFeature::MonthEnd
        | CalendarFeature::DayOfYear
        | CalendarFeature::ElapsedIndex
        | CalendarFeature::ElapsedPhase(_) => true,
    }
}

fn calendar_feature_value(
    feature: &CalendarFeature,
    timestamp: NaiveDateTime,
    prior_len: usize,
) -> f64 {
    match feature {
        CalendarFeature::DayOfWeek => f64::from(timestamp.weekday().num_days_from_monday()),
        CalendarFeature::DayOfWeekSin => {
            cyclic_sin(f64::from(timestamp.weekday().num_days_from_monday()), 7.0)
        }
        CalendarFeature::DayOfWeekCos => {
            cyclic_cos(f64::from(timestamp.weekday().num_days_from_monday()), 7.0)
        }
        CalendarFeature::Month => f64::from(timestamp.month()),
        CalendarFeature::MonthSin => cyclic_sin(f64::from(timestamp.month0()), 12.0),
        CalendarFeature::MonthCos => cyclic_cos(f64::from(timestamp.month0()), 12.0),
        CalendarFeature::Day => f64::from(timestamp.day()),
        CalendarFeature::DaySin => cyclic_sin(f64::from(timestamp.day0()), 31.0),
        CalendarFeature::DayCos => cyclic_cos(f64::from(timestamp.day0()), 31.0),
        CalendarFeature::MonthStart => {
            if timestamp.day() <= 3 {
                1.0
            } else {
                0.0
            }
        }
        CalendarFeature::MonthMiddle => {
            if (14..=16).contains(&timestamp.day()) {
                1.0
            } else {
                0.0
            }
        }
        CalendarFeature::MonthEnd => {
            let Some(days_in_month) = days_in_month(timestamp) else {
                return 0.0;
            };
            if timestamp.day() + 2 >= days_in_month {
                1.0
            } else {
                0.0
            }
        }
        CalendarFeature::DayOfYear => f64::from(timestamp.ordinal()),
        CalendarFeature::ElapsedIndex => prior_len as f64,
        CalendarFeature::ElapsedPhase(period) => (prior_len % *period) as f64,
    }
}

fn cyclic_sin(position: f64, period: f64) -> f64 {
    (std::f64::consts::TAU * position / period).sin()
}

fn cyclic_cos(position: f64, period: f64) -> f64 {
    (std::f64::consts::TAU * position / period).cos()
}

fn days_in_month(timestamp: NaiveDateTime) -> Option<u32> {
    let (next_year, next_month) = if timestamp.month() == 12 {
        (timestamp.year() + 1, 1)
    } else {
        (timestamp.year(), timestamp.month() + 1)
    };
    chrono::NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .and_then(|first_next| first_next.pred_opt())
        .map(|last_this| last_this.day())
}

