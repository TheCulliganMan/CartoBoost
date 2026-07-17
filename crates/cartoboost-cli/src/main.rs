use cartoboost_core::loss::{LossConfig, QuantileLossConfig};
use cartoboost_core::tree::{FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::{Booster, BoosterConfig, Dataset, Model as CoreModel};
use cartoboost_neural::{
    available_backends, backend_supports_operation, backend_workload_decision, select_backend_for,
    BackendOperation,
};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

type CliResult<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(err) = run() {
        eprintln!("cartoboost: {err}");
        std::process::exit(1);
    }
}

fn run() -> CliResult<()> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return Ok(());
    };

    if command == "-h" || command == "--help" {
        print_help();
        return Ok(());
    }

    let opts = parse_options(args.collect())?;
    validate_options(command.as_str(), &opts)?;
    if opts.contains_key("help") {
        print_help();
        return Ok(());
    }

    match command.as_str() {
        "train" => train(opts),
        "predict" => predict(opts),
        "eval" => evaluate(opts),
        "inspect" => inspect(opts),
        "accelerators" => accelerators(opts),
        other => Err(format!("unknown command '{other}'").into()),
    }
}

fn print_help() {
    println!(
        "cartoboost <command> [options]\n\nBackends: auto|cpu|cuda|rocm|hip|metal|directml|dml|webgpu\n\nCommands:\n  train         --data <csv> [--backend <name>] [--config <toml>] [--model-out <path>] [--output json|csv]\n  predict       --model <path> --input <csv> [--backend <name>] [--predictions-out <path>] [--output json|csv]\n  eval          --model <path> --data <csv> [--backend <name>] [--output json|csv]\n  inspect       [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]\n  accelerators  [--backend <name> --operation <kernel> --workload-size <n> --minimum-accelerated-size <n>] [--output json|csv]"
    );
}

fn parse_options(raw: Vec<String>) -> CliResult<BTreeMap<String, String>> {
    let mut opts = BTreeMap::new();
    let mut i = 0;
    while i < raw.len() {
        let key = &raw[i];
        if !key.starts_with("--") {
            return Err(format!("expected option, got '{key}'").into());
        }
        let name = key.trim_start_matches("--").to_string();
        if name == "help" {
            opts.insert(name, "true".to_string());
            i += 1;
            continue;
        }
        let Some(value) = raw.get(i + 1) else {
            return Err(format!("missing value for '{key}'").into());
        };
        opts.insert(name, value.clone());
        i += 2;
    }
    Ok(opts)
}

fn validate_options(command: &str, opts: &BTreeMap<String, String>) -> CliResult<()> {
    let allowed = match command {
        "train" => &["backend", "config", "data", "help", "model-out", "output"][..],
        "predict" => &[
            "backend",
            "help",
            "input",
            "model",
            "output",
            "predictions-out",
        ][..],
        "eval" => &["backend", "data", "help", "model", "output"][..],
        "inspect" => &["config", "data", "help", "model", "output"][..],
        "accelerators" => &[
            "backend",
            "help",
            "minimum-accelerated-size",
            "operation",
            "output",
            "workload-size",
        ][..],
        _ => return Ok(()),
    };
    for key in opts.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown option '--{key}' for command '{command}'").into());
        }
    }
    Ok(())
}

fn accelerators(opts: BTreeMap<String, String>) -> CliResult<()> {
    let output = output_format(&opts)?;
    if let Some(operation_name) = opts.get("operation") {
        let operation = BackendOperation::ALL
            .into_iter()
            .find(|operation| operation.as_str() == operation_name)
            .ok_or_else(|| format!("unknown accelerator operation '{operation_name}'"))?;
        let workload_size = opts
            .get("workload-size")
            .ok_or("--workload-size is required with --operation")?
            .parse::<usize>()?;
        let minimum = opts
            .get("minimum-accelerated-size")
            .ok_or("--minimum-accelerated-size is required with --operation")?
            .parse::<usize>()?;
        let selection = select_backend_for(opts.get("backend").map(String::as_str), operation)?;
        let decision = backend_workload_decision(&selection, operation, workload_size, minimum);
        if output == "csv" {
            println!(
                "requested,selected,executed,operation,workload_size,minimum_accelerated_size,accelerated,fallback_reason"
            );
            println!(
                "{},{},{},{},{},{},{},{}",
                csv_cell(&decision.requested),
                csv_cell(&decision.selected),
                csv_cell(&decision.executed),
                csv_cell(&decision.operation),
                decision.workload_size,
                decision.minimum_accelerated_size,
                decision.accelerated,
                csv_cell(decision.fallback_reason.as_deref().unwrap_or(""))
            );
        } else {
            println!(
                "{{\"requested\":\"{}\",\"selected\":\"{}\",\"executed\":\"{}\",\"operation\":\"{}\",\"workload_size\":{},\"minimum_accelerated_size\":{},\"accelerated\":{},\"fallback_reason\":{}}}",
                json_escape(&decision.requested),
                json_escape(&decision.selected),
                json_escape(&decision.executed),
                json_escape(&decision.operation),
                decision.workload_size,
                decision.minimum_accelerated_size,
                decision.accelerated,
                decision.fallback_reason.as_ref().map_or_else(
                    || "null".to_string(),
                    |reason| format!("\"{}\"", json_escape(reason))
                )
            );
        }
        return Ok(());
    }
    let available = available_backends();
    if output == "csv" {
        println!("backend,available,operations");
        for backend in ["cpu", "cuda", "rocm", "metal", "directml", "webgpu"] {
            let operations = BackendOperation::ALL
                .into_iter()
                .filter(|operation| backend_supports_operation(backend, *operation))
                .map(BackendOperation::as_str)
                .collect::<Vec<_>>()
                .join(";");
            println!(
                "{backend},{},{}",
                available.iter().any(|candidate| candidate == backend),
                operations
            );
        }
        return Ok(());
    }
    let rows = ["cpu", "cuda", "rocm", "metal", "directml", "webgpu"]
        .into_iter()
        .map(|backend| {
            let operations = BackendOperation::ALL
                .into_iter()
                .filter(|operation| backend_supports_operation(backend, *operation))
                .map(|operation| format!("\"{}\"", operation.as_str()))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"backend\":\"{backend}\",\"available\":{},\"operations\":[{operations}]}}",
                available.iter().any(|candidate| candidate == backend)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"backends\":[{rows}]}}");
    Ok(())
}

fn train(opts: BTreeMap<String, String>) -> CliResult<()> {
    let data_path = required(&opts, "data")?;
    let output = output_format(&opts)?;
    let rows = read_csv(data_path)?;
    let cfg = opts
        .get("config")
        .map(|path| read_config(path))
        .transpose()?
        .unwrap_or_default();
    let target = cfg
        .target
        .clone()
        .or_else(|| rows.headers.last().cloned())
        .unwrap_or_else(|| "target".to_string());
    let (dataset, y, feature_count) = numeric_dataset_and_target(&rows, &target)?;
    let model_path = opts
        .get("model-out")
        .cloned()
        .unwrap_or_else(|| "cartoboost-model.json".to_string());
    let backend = opts.get("backend").map(String::as_str).unwrap_or("cpu");
    let booster = Booster::new_with_backend(cfg.booster_config()?, Some(backend))?;
    let selected_backend = booster.backend().selected.clone();
    let mut model = booster.fit(&dataset, &y, None)?;
    model.target_name = Some(target);
    model.save(&model_path)?;

    match output.as_str() {
        "csv" => println!(
            "ok,command,rows,features,model_path,backend\ntrue,train,{},{},{},{}",
            rows.records.len(),
            feature_count,
            csv_cell(&model_path),
            csv_cell(&selected_backend)
        ),
        _ => println!(
            "{{\"ok\":true,\"command\":\"train\",\"rows\":{},\"features\":{},\"model_path\":\"{}\",\"trees\":{},\"backend\":\"{}\"}}",
            rows.records.len(),
            feature_count,
            json_escape(&model_path),
            model.trees.len(),
            json_escape(&selected_backend)
        ),
    }
    Ok(())
}

fn predict(opts: BTreeMap<String, String>) -> CliResult<()> {
    let model_path = required(&opts, "model")?;
    let input_path = required(&opts, "input")?;
    let output = output_format(&opts)?;
    let rows = read_csv(input_path)?;
    if let Ok(model) = CoreModel::load(model_path) {
        let dataset = numeric_dataset_without_target(&rows, model.feature_count)?;
        let requested_backend = opts.get("backend").map(String::as_str).unwrap_or("cpu");
        let selected_backend =
            select_backend_for(Some(requested_backend), BackendOperation::Dense)?;
        let predictions =
            model.try_predict_with_backend(&dataset, Some(&selected_backend.selected))?;
        let mut csv = String::from("row,prediction\n");
        for (idx, prediction) in predictions.iter().enumerate() {
            csv.push_str(&format!("{idx},{}\n", format_float(*prediction)));
        }
        if let Some(path) = opts.get("predictions-out") {
            fs::write(path, &csv)?;
        }

        match output.as_str() {
            "csv" => print!("{csv}"),
            _ => println!(
                "{{\"ok\":true,\"command\":\"predict\",\"rows\":{},\"backend\":\"{}\",\"predictions_out\":{}}}",
                predictions.len(),
                json_escape(&selected_backend.selected),
                optional_json_string(opts.get("predictions-out"))
            ),
        }
        return Ok(());
    }

    let model = read_model(model_path)?;
    ensure_feature_count(&rows, model.feature_count)?;
    let mut csv = String::from("row,prediction\n");
    for idx in 0..rows.records.len() {
        csv.push_str(&format!(
            "{},{}\n",
            idx,
            csv_cell(&model.baseline.as_string())
        ));
    }
    if let Some(path) = opts.get("predictions-out") {
        fs::write(path, &csv)?;
    }

    match output.as_str() {
        "csv" => print!("{csv}"),
        _ => println!(
            "{{\"ok\":true,\"command\":\"predict\",\"rows\":{},\"prediction\":{},\"predictions_out\":{}}}",
            rows.records.len(),
            model.baseline.to_json_value(),
            optional_json_string(opts.get("predictions-out"))
        ),
    }
    Ok(())
}

fn evaluate(opts: BTreeMap<String, String>) -> CliResult<()> {
    let model_path = required(&opts, "model")?;
    let data_path = required(&opts, "data")?;
    let output = output_format(&opts)?;
    let rows = read_csv(data_path)?;
    if let Ok(model) = CoreModel::load(model_path) {
        let target = model
            .target_name
            .clone()
            .or_else(|| rows.headers.last().cloned())
            .unwrap_or_else(|| "target".to_string());
        let (dataset, y, _) = numeric_dataset_and_target(&rows, &target)?;
        let requested_backend = opts.get("backend").map(String::as_str).unwrap_or("cpu");
        let selected_backend =
            select_backend_for(Some(requested_backend), BackendOperation::Dense)?;
        let predictions =
            model.try_predict_with_backend(&dataset, Some(&selected_backend.selected))?;
        let mae = y
            .iter()
            .zip(&predictions)
            .map(|(actual, prediction)| (actual - prediction).abs())
            .sum::<f64>()
            / y.len().max(1) as f64;

        match output.as_str() {
            "csv" => println!(
                "ok,command,rows,target,mae,backend\ntrue,eval,{},{},{},{}",
                rows.records.len(),
                csv_cell(&target),
                format_float(mae),
                csv_cell(&selected_backend.selected)
            ),
            _ => println!(
                "{{\"ok\":true,\"command\":\"eval\",\"rows\":{},\"target\":\"{}\",\"mae\":{},\"backend\":\"{}\"}}",
                rows.records.len(),
                json_escape(&target),
                format_float(mae),
                json_escape(&selected_backend.selected)
            ),
        }
        return Ok(());
    }

    let model = read_model(model_path)?;
    let target_idx = rows.headers.iter().position(|name| name == &model.target);
    let mae = target_idx.and_then(|idx| mean_absolute_error(&rows, idx, model.baseline.numeric()));

    match output.as_str() {
        "csv" => println!(
            "ok,command,rows,target,mae\ntrue,eval,{},{},{}",
            rows.records.len(),
            csv_cell(&model.target),
            mae.map(format_float).unwrap_or_else(|| "".to_string())
        ),
        _ => println!(
            "{{\"ok\":true,\"command\":\"eval\",\"rows\":{},\"target\":\"{}\",\"mae\":{}}}",
            rows.records.len(),
            json_escape(&model.target),
            mae.map(format_float).unwrap_or_else(|| "null".to_string())
        ),
    }
    Ok(())
}

fn inspect(opts: BTreeMap<String, String>) -> CliResult<()> {
    let output = output_format(&opts)?;
    let model = opts.get("model").map(|path| read_model(path)).transpose()?;
    let config = opts
        .get("config")
        .map(|path| read_config(path))
        .transpose()?;
    let data = opts.get("data").map(|path| read_csv(path)).transpose()?;

    match output.as_str() {
        "csv" => {
            println!("kind,path,rows,columns,target");
            if let Some(path) = opts.get("model") {
                let Some(model) = &model else { unreachable!() };
                println!(
                    "model,{},{},{},{}",
                    csv_cell(path),
                    model.rows,
                    model.feature_count + 1,
                    csv_cell(&model.target)
                );
            }
            if let Some(path) = opts.get("config") {
                let target = config
                    .as_ref()
                    .and_then(|cfg| cfg.target.as_ref())
                    .cloned()
                    .unwrap_or_default();
                println!("config,{},,,{}", csv_cell(path), csv_cell(&target));
            }
            if let Some(path) = opts.get("data") {
                let Some(data) = &data else { unreachable!() };
                println!(
                    "data,{},{},{},",
                    csv_cell(path),
                    data.records.len(),
                    data.headers.len()
                );
            }
        }
        _ => {
            let model_json = model
                .as_ref()
                .map(Model::summary_json)
                .unwrap_or_else(|| "null".to_string());
            let config_json = config
                .as_ref()
                .map(Config::summary_json)
                .unwrap_or_else(|| "null".to_string());
            let data_json = data
                .as_ref()
                .map(CsvData::summary_json)
                .unwrap_or_else(|| "null".to_string());
            println!("{{\"ok\":true,\"command\":\"inspect\",\"model\":{model_json},\"config\":{config_json},\"data\":{data_json}}}");
        }
    }
    Ok(())
}

fn required<'a>(opts: &'a BTreeMap<String, String>, name: &str) -> CliResult<&'a str> {
    opts.get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required option '--{name}'").into())
}

fn output_format(opts: &BTreeMap<String, String>) -> CliResult<String> {
    let output = opts
        .get("output")
        .cloned()
        .unwrap_or_else(|| "json".to_string());
    match output.as_str() {
        "json" | "csv" => Ok(output),
        other => {
            Err(format!("unsupported output format '{other}', expected 'json' or 'csv'").into())
        }
    }
}

#[derive(Default)]
struct Config {
    target: Option<String>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    min_gain: Option<f64>,
    loss: Option<String>,
    quantile_alpha: Option<f64>,
    splitter: Option<String>,
    leaf_predictor: Option<String>,
    fuzzy: Option<bool>,
    fuzzy_bandwidth: Option<f64>,
    fuzzy_kernel: Option<String>,
    l2_regularization: Option<f64>,
    monotonic_constraints: Option<Vec<i8>>,
}

impl Config {
    fn summary_json(&self) -> String {
        format!(
            "{{\"target\":{}}}",
            optional_json_string(self.target.as_ref())
        )
    }

    fn booster_config(&self) -> CliResult<BoosterConfig> {
        let defaults = BoosterConfig::default();
        Ok(BoosterConfig {
            n_estimators: self.n_estimators.unwrap_or(defaults.n_estimators),
            learning_rate: self.learning_rate.unwrap_or(defaults.learning_rate),
            max_depth: self.max_depth.unwrap_or(defaults.max_depth),
            min_samples_leaf: self.min_samples_leaf.unwrap_or(defaults.min_samples_leaf),
            min_gain: self.min_gain.unwrap_or(defaults.min_gain),
            loss: self
                .loss
                .as_deref()
                .map(|loss| cli_loss(loss, self.quantile_alpha.unwrap_or(0.5)))
                .transpose()?
                .unwrap_or(defaults.loss),
            splitters: self
                .splitter
                .as_deref()
                .map(cli_splitters)
                .transpose()?
                .unwrap_or(defaults.splitters),
            leaf_predictor: self
                .leaf_predictor
                .as_deref()
                .map(cli_leaf_predictor)
                .transpose()?
                .unwrap_or(defaults.leaf_predictor),
            linear_leaf_features: defaults.linear_leaf_features,
            linear_lambda_l2: self.l2_regularization.unwrap_or(defaults.linear_lambda_l2),
            constant_lambda_l2: defaults.constant_lambda_l2,
            fuzzy: self.fuzzy.unwrap_or(defaults.fuzzy),
            fuzzy_bandwidth: self.fuzzy_bandwidth.unwrap_or(defaults.fuzzy_bandwidth),
            fuzzy_kernel: self
                .fuzzy_kernel
                .as_deref()
                .map(cli_fuzzy_kernel)
                .transpose()?
                .unwrap_or(defaults.fuzzy_kernel),
            monotonic_constraints: self
                .monotonic_constraints
                .clone()
                .unwrap_or(defaults.monotonic_constraints),
            interaction_constraints: defaults.interaction_constraints,
            graph_split_regularization: defaults.graph_split_regularization,
            graph_leaf_smoothing: defaults.graph_leaf_smoothing,
        })
    }
}

fn read_config(path: &str) -> CliResult<Config> {
    let text = fs::read_to_string(path)?;
    let mut config = Config::default();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_number = line_idx + 1;
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("invalid config line {line_number}: expected key = value").into());
        };
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() {
            return Err(format!("invalid config line {line_number}: empty key").into());
        }
        if value.is_empty() {
            return Err(format!(
                "invalid config value for '{key}' on line {line_number}: empty value"
            )
            .into());
        }
        match key {
            "target" | "target_column" => {
                config.target = Some(parse_config_string(key, value, line_number)?)
            }
            "n_estimators" => {
                config.n_estimators = Some(parse_config_value(key, value, line_number)?)
            }
            "learning_rate" => {
                config.learning_rate = Some(parse_config_value(key, value, line_number)?)
            }
            "max_depth" => config.max_depth = Some(parse_config_value(key, value, line_number)?),
            "min_samples_leaf" => {
                config.min_samples_leaf = Some(parse_config_value(key, value, line_number)?)
            }
            "min_gain" => config.min_gain = Some(parse_config_value(key, value, line_number)?),
            "loss" => {
                let loss = parse_config_string(key, value, line_number)?;
                cli_loss(&loss, config.quantile_alpha.unwrap_or(0.5))?;
                config.loss = Some(loss);
            }
            "quantile_alpha" => {
                config.quantile_alpha = Some(parse_config_value(key, value, line_number)?)
            }
            "splitter" | "splitters" => {
                let splitters = parse_config_string(key, value, line_number)?;
                cli_splitters(&splitters)?;
                config.splitter = Some(splitters);
            }
            "leaf_predictor" => {
                let leaf_predictor = parse_config_string(key, value, line_number)?;
                cli_leaf_predictor(&leaf_predictor)?;
                config.leaf_predictor = Some(leaf_predictor);
            }
            "fuzzy" => config.fuzzy = Some(parse_config_value(key, value, line_number)?),
            "fuzzy_bandwidth" => {
                config.fuzzy_bandwidth = Some(parse_config_value(key, value, line_number)?)
            }
            "fuzzy_kernel" => {
                let fuzzy_kernel = parse_config_string(key, value, line_number)?;
                cli_fuzzy_kernel(&fuzzy_kernel)?;
                config.fuzzy_kernel = Some(fuzzy_kernel);
            }
            "l2_regularization" => {
                config.l2_regularization = Some(parse_config_value(key, value, line_number)?)
            }
            "monotonic_constraints" => {
                config.monotonic_constraints =
                    Some(parse_monotonic_constraints(value, line_number)?)
            }
            _ => {
                return Err(format!("unknown config key '{key}' on line {line_number}").into());
            }
        }
    }
    Ok(config)
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_quote = None;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_quote == Some('"') => escaped = true,
            '"' | '\'' if in_quote == Some(ch) => in_quote = None,
            '"' | '\'' if in_quote.is_none() => in_quote = Some(ch),
            '#' if in_quote.is_none() => return &line[..idx],
            _ => {}
        }
    }
    line
}

fn parse_config_string(key: &str, value: &str, line_number: usize) -> CliResult<String> {
    let trimmed = value.trim();
    if let Some(quote) = trimmed
        .chars()
        .next()
        .filter(|ch| *ch == '"' || *ch == '\'')
    {
        if !trimmed.ends_with(quote) || trimmed.len() < 2 {
            return Err(format!(
                "invalid config value for '{key}' on line {line_number}: unterminated string"
            )
            .into());
        }
        return Ok(trimmed[1..trimmed.len() - 1].to_string());
    }
    if trimmed.contains(char::is_whitespace) {
        return Err(format!(
            "invalid config value for '{key}' on line {line_number}: strings with whitespace must be quoted"
        )
        .into());
    }
    Ok(trimmed.to_string())
}

fn parse_config_value<T>(key: &str, value: &str, line_number: usize) -> CliResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = parse_config_string(key, value, line_number)?;
    value.parse::<T>().map_err(|err| {
        format!("invalid config value for '{key}' on line {line_number}: {err}").into()
    })
}

fn cli_splitters(value: &str) -> CliResult<Vec<SplitterKind>> {
    let mut splitters = Vec::new();
    for raw_name in value.split(',') {
        let name = raw_name.trim();
        if name.is_empty() {
            return Err("splitter list contains an empty value".into());
        }
        let splitter = match name {
            "auto" => SplitterKind::Auto,
            "axis" => SplitterKind::Axis,
            "axis_histogram" | "axis_hist" | "histogram" => {
                SplitterKind::AxisHistogram { bins: 64 }
            }
            "diagonal_2d" | "diagonal2d" => SplitterKind::Diagonal2D,
            "gaussian_2d" | "gaussian2d" | "radial" => SplitterKind::Gaussian2D,
            "periodic_time" | "periodic_24" => SplitterKind::Periodic { period: 24.0 },
            "sparse_set" | "sparse" => SplitterKind::SparseSet,
            _ => {
                if let Some(bins) = name
                    .strip_prefix("axis_histogram:")
                    .or_else(|| name.strip_prefix("axis_hist:"))
                    .and_then(|bins| bins.parse::<usize>().ok())
                    .filter(|bins| *bins >= 2)
                {
                    SplitterKind::AxisHistogram { bins }
                } else {
                    return Err(format!("unknown splitter '{name}'").into());
                }
            }
        };
        splitters.push(splitter);
    }
    if splitters.is_empty() {
        Err("splitter list must not be empty".into())
    } else {
        Ok(splitters)
    }
}

fn cli_leaf_predictor(value: &str) -> CliResult<LeafPredictorKind> {
    match value {
        "constant" => Ok(LeafPredictorKind::Constant),
        "linear" => Ok(LeafPredictorKind::Linear),
        other => Err(format!("unknown leaf_predictor '{other}'").into()),
    }
}

fn cli_loss(value: &str, quantile_alpha: f64) -> CliResult<LossConfig> {
    match value {
        "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "mae" | "absolute_error" | "least_absolute_deviation" | "lad" => Ok(LossConfig::L1),
        "quantile" | "pinball" => {
            if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
                return Err("quantile_alpha must be finite and in (0, 1)".into());
            }
            Ok(LossConfig::Quantile(QuantileLossConfig {
                alpha: quantile_alpha,
            }))
        }
        other => Err(format!("unknown loss '{other}'").into()),
    }
}

fn cli_fuzzy_kernel(value: &str) -> CliResult<FuzzyKernel> {
    match value {
        "linear" | "triangular" => Ok(FuzzyKernel::Linear),
        "gaussian" => Ok(FuzzyKernel::Gaussian),
        "exponential" => Ok(FuzzyKernel::Exponential),
        "bisquare" => Ok(FuzzyKernel::Bisquare),
        "epanechnikov" => Ok(FuzzyKernel::Epanechnikov),
        "tricube" => Ok(FuzzyKernel::Tricube),
        other => Err(format!("unknown fuzzy_kernel '{other}'").into()),
    }
}

fn parse_monotonic_constraints(value: &str, line_number: usize) -> CliResult<Vec<i8>> {
    let value = parse_config_string("monotonic_constraints", value, line_number)?;
    let mut constraints = Vec::new();
    for raw in value.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            return Err("monotonic_constraints contains an empty value".into());
        }
        let constraint = item.parse::<i8>().map_err(|err| {
            format!("invalid monotonic_constraints value on line {line_number}: {err}")
        })?;
        if !matches!(constraint, -1..=1) {
            return Err("monotonic_constraints values must be -1, 0, or 1".into());
        }
        constraints.push(constraint);
    }
    Ok(constraints)
}

struct CsvData {
    headers: Vec<String>,
    records: Vec<Vec<String>>,
}

impl CsvData {
    fn summary_json(&self) -> String {
        format!(
            "{{\"rows\":{},\"columns\":{}}}",
            self.records.len(),
            self.headers.len()
        )
    }
}

fn read_csv(path: &str) -> CliResult<CsvData> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let headers = lines
        .next()
        .map(parse_csv_line)
        .ok_or_else(|| format!("CSV file '{}' is empty", Path::new(path).display()))?;
    let records = lines.map(parse_csv_line).collect::<Vec<_>>();
    let records = records
        .into_iter()
        .enumerate()
        .map(|(idx, record)| {
            if record.len() != headers.len() {
                Err(format!(
                    "CSV row {} has {} columns but header has {}",
                    idx + 2,
                    record.len(),
                    headers.len()
                )
                .into())
            } else {
                Ok(record)
            }
        })
        .collect::<CliResult<Vec<_>>>()?;
    Ok(CsvData { headers, records })
}

fn numeric_dataset_and_target(
    rows: &CsvData,
    target: &str,
) -> CliResult<(Dataset, Vec<f64>, usize)> {
    let target_idx = rows
        .headers
        .iter()
        .position(|name| name == target)
        .ok_or_else(|| format!("target column '{target}' not found"))?;
    let feature_indices = rows
        .headers
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| (idx != target_idx).then_some(idx))
        .collect::<Vec<_>>();
    let mut x = Vec::with_capacity(rows.records.len());
    let mut y = Vec::with_capacity(rows.records.len());
    for record in &rows.records {
        x.push(parse_numeric_row(record, &feature_indices)?);
        y.push(parse_numeric_cell(record, target_idx)?);
    }
    let feature_count = feature_indices.len();
    Ok((Dataset::from_rows(x)?, y, feature_count))
}

fn numeric_dataset_without_target(rows: &CsvData, feature_count: usize) -> CliResult<Dataset> {
    ensure_feature_count(rows, feature_count)?;
    let feature_indices = (0..feature_count).collect::<Vec<_>>();
    let x = rows
        .records
        .iter()
        .map(|record| parse_numeric_row(record, &feature_indices))
        .collect::<CliResult<Vec<_>>>()?;
    Ok(Dataset::from_rows(x)?)
}

fn ensure_feature_count(rows: &CsvData, feature_count: usize) -> CliResult<()> {
    if rows.headers.len() != feature_count {
        return Err(format!(
            "input has {} feature columns but model expects {feature_count}",
            rows.headers.len()
        )
        .into());
    }
    Ok(())
}

fn parse_numeric_row(record: &[String], indices: &[usize]) -> CliResult<Vec<f64>> {
    indices
        .iter()
        .map(|idx| parse_numeric_cell(record, *idx))
        .collect()
}

fn parse_numeric_cell(record: &[String], idx: usize) -> CliResult<f64> {
    record
        .get(idx)
        .ok_or_else(|| format!("row is missing column index {idx}").into())
        .and_then(|value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("failed to parse numeric value '{value}': {err}").into())
        })
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cell.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                values.push(cell.trim().to_string());
                cell.clear();
            }
            _ => cell.push(ch),
        }
    }
    values.push(cell.trim().to_string());
    values
}

#[derive(Clone)]
enum Baseline {
    Numeric(f64),
    Text(String),
}

impl Baseline {
    fn as_string(&self) -> String {
        match self {
            Baseline::Numeric(value) => format_float(*value),
            Baseline::Text(value) => value.clone(),
        }
    }

    fn numeric(&self) -> Option<f64> {
        match self {
            Baseline::Numeric(value) => Some(*value),
            Baseline::Text(_) => None,
        }
    }

    fn to_json_value(&self) -> String {
        match self {
            Baseline::Numeric(value) => format_float(*value),
            Baseline::Text(value) => format!("\"{}\"", json_escape(value)),
        }
    }
}

struct Model {
    target: String,
    rows: usize,
    feature_count: usize,
    baseline: Baseline,
}

impl Model {
    fn summary_json(&self) -> String {
        format!(
            "{{\"model_type\":\"constant_baseline\",\"rows\":{},\"features\":{},\"target\":\"{}\",\"baseline\":{}}}",
            self.rows,
            self.feature_count,
            json_escape(&self.target),
            self.baseline.to_json_value()
        )
    }
}

fn mean_absolute_error(rows: &CsvData, target_idx: usize, prediction: Option<f64>) -> Option<f64> {
    let prediction = prediction?;
    let mut total = 0.0;
    let mut count = 0usize;
    for record in &rows.records {
        if let Some(actual) = record
            .get(target_idx)
            .and_then(|value| value.parse::<f64>().ok())
        {
            total += (actual - prediction).abs();
            count += 1;
        }
    }
    (count > 0).then_some(total / count as f64)
}

fn read_model(path: &str) -> CliResult<Model> {
    let text = fs::read_to_string(path)?;
    let target = json_string_field(&text, "target").unwrap_or_else(|| "target".to_string());
    let rows = json_usize_field(&text, "rows").unwrap_or(0);
    let feature_count = json_usize_field(&text, "features").unwrap_or(0);
    let baseline = if let Some(value) = json_number_field(&text, "baseline") {
        Baseline::Numeric(value)
    } else {
        Baseline::Text(json_string_field(&text, "baseline").unwrap_or_default())
    };
    Ok(Model {
        target,
        rows,
        feature_count,
        baseline,
    })
}

fn json_string_field(text: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\"");
    let after_key = text.split(&key).nth(1)?;
    let after_colon = after_key.split_once(':')?.1.trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for ch in after_colon[1..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_usize_field(text: &str, field: &str) -> Option<usize> {
    json_number_token(text, field)?.parse().ok()
}

fn json_number_field(text: &str, field: &str) -> Option<f64> {
    json_number_token(text, field)?.parse().ok()
}

fn json_number_token<'a>(text: &'a str, field: &str) -> Option<&'a str> {
    let key = format!("\"{field}\"");
    let after_key = text.split(&key).nth(1)?;
    let after_colon = after_key.split_once(':')?.1.trim_start();
    let end = after_colon
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.' || ch == '-'))
        .unwrap_or(after_colon.len());
    (end > 0).then_some(&after_colon[..end])
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn optional_json_string(value: Option<&String>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn csv_cell(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn format_float(value: f64) -> String {
    let formatted = format!("{value:.6}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}
