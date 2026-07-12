/* @ts-self-types="./cartoboost_wasm.d.ts" */

/**
 * @returns {any}
 */
export function availableDeepBackends() {
    const ret = wasm.availableDeepBackends();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @returns {any}
 */
export function availableForecastModels() {
    const ret = wasm.availableForecastModels();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} candidates
 * @param {number} temperature
 * @param {string | null} [monotone_candidate_value]
 * @returns {any}
 */
export function deepChoiceSetTransformerReport(candidates, temperature, monotone_candidate_value) {
    var ptr0 = isLikeNone(monotone_candidate_value) ? 0 : passStringToWasm0(monotone_candidate_value, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.deepChoiceSetTransformerReport(candidates, temperature, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} hidden
 * @param {Float64Array} residuals
 * @param {Float64Array} quantiles
 * @param {number} sample_count
 * @returns {any}
 */
export function deepConditionalFlowFit(hidden, residuals, quantiles, sample_count) {
    const ptr0 = passArrayF64ToWasm0(residuals, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(quantiles, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.deepConditionalFlowFit(hidden, ptr0, len0, ptr1, len1, sample_count);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {any} hidden
 * @param {Float64Array | null} [actual]
 * @returns {any}
 */
export function deepConditionalFlowPredict(artifact, hidden, actual) {
    var ptr0 = isLikeNone(actual) ? 0 : passArrayF64ToWasm0(actual, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.deepConditionalFlowPredict(artifact, hidden, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} candidates
 * @param {string} objective
 * @param {any} constraints
 * @param {string} fallback
 * @returns {any}
 */
export function deepConstrainedDecisionSelect(candidates, objective, constraints, fallback) {
    const ptr0 = passStringToWasm0(objective, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(fallback, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.deepConstrainedDecisionSelect(candidates, ptr0, len0, constraints, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} point_forecast
 * @param {any} edges
 * @param {number} scenario_count
 * @param {number} diffusion_steps
 * @param {number} shock_scale
 * @returns {any}
 */
export function deepDiffusionScenarioGenerate(point_forecast, edges, scenario_count, diffusion_steps, shock_scale) {
    const ret = wasm.deepDiffusionScenarioGenerate(point_forecast, edges, scenario_count, diffusion_steps, shock_scale);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} rows
 * @returns {any}
 */
export function deepDirectionalPairPredict(rows) {
    const ret = wasm.deepDirectionalPairPredict(rows);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} features
 * @param {Float64Array} labels
 * @param {string | null} [backend]
 * @returns {any}
 */
export function deepEventOutcomeFit(features, labels, backend) {
    const ptr0 = passArrayF64ToWasm0(labels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = isLikeNone(backend) ? 0 : passStringToWasm0(backend, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    const ret = wasm.deepEventOutcomeFit(features, ptr0, len0, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {any} features
 * @returns {any}
 */
export function deepEventOutcomePredict(artifact, features) {
    const ret = wasm.deepEventOutcomePredict(artifact, features);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Predicts event probabilities with the artifact's hidden layer dispatched on
 * WebGPU. The export is asynchronous because browser GPU work is asynchronous.
 * @param {any} artifact
 * @param {any} features
 * @returns {Promise<any>}
 */
export function deepEventOutcomePredictWebgpu(artifact, features) {
    const ret = wasm.deepEventOutcomePredictWebgpu(artifact, features);
    return ret;
}

/**
 * @param {any} field_values
 * @param {any} coordinates
 * @param {any} edges
 * @param {any} exogenous_fields
 * @param {number} smoothing
 * @param {number} coordinate_scale
 * @returns {any}
 */
export function deepGraphNeuralOperatorPredict(field_values, coordinates, edges, exogenous_fields, smoothing, coordinate_scale) {
    const ret = wasm.deepGraphNeuralOperatorPredict(field_values, coordinates, edges, exogenous_fields, smoothing, coordinate_scale);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @returns {any}
 */
export function deepNeuralOperatorSyntheticBenchmark() {
    const ret = wasm.deepNeuralOperatorSyntheticBenchmark();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} features
 * @param {Float64Array} target
 * @returns {any}
 */
export function deepRegimeMoeReport(features, target) {
    const ptr0 = passArrayF64ToWasm0(target, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.deepRegimeMoeReport(features, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} rows
 * @param {string} response_type
 * @param {string | null} [monotone]
 * @param {string | null} [backend]
 * @returns {any}
 */
export function deepResponseCurveFit(rows, response_type, monotone, backend) {
    const ptr0 = passStringToWasm0(response_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = isLikeNone(monotone) ? 0 : passStringToWasm0(monotone, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    var ptr2 = isLikeNone(backend) ? 0 : passStringToWasm0(backend, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len2 = WASM_VECTOR_LEN;
    const ret = wasm.deepResponseCurveFit(rows, ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {any} rows
 * @returns {any}
 */
export function deepResponseCurvePredict(artifact, rows) {
    const ret = wasm.deepResponseCurvePredict(artifact, rows);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {any} rows
 * @returns {Promise<any>}
 */
export function deepResponseCurvePredictWebgpu(artifact, rows) {
    const ret = wasm.deepResponseCurvePredictWebgpu(artifact, rows);
    return ret;
}

/**
 * @param {any} rows
 * @param {string | null} [backend]
 * @returns {any}
 */
export function deepServiceResidualFit(rows, backend) {
    var ptr0 = isLikeNone(backend) ? 0 : passStringToWasm0(backend, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.deepServiceResidualFit(rows, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {any} rows
 * @returns {any}
 */
export function deepServiceResidualPredict(artifact, rows) {
    const ret = wasm.deepServiceResidualPredict(artifact, rows);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {any} rows
 * @returns {Promise<any>}
 */
export function deepServiceResidualPredictWebgpu(artifact, rows) {
    const ret = wasm.deepServiceResidualPredictWebgpu(artifact, rows);
    return ret;
}

/**
 * @param {any} y
 * @param {number} lookback
 * @param {number} horizon
 * @returns {any}
 */
export function deepTemporalEntityFit(y, lookback, horizon) {
    const ret = wasm.deepTemporalEntityFit(y, lookback, horizon);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} artifact
 * @param {number} horizon
 * @returns {any}
 */
export function deepTemporalEntityPredict(artifact, horizon) {
    const ret = wasm.deepTemporalEntityPredict(artifact, horizon);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function fitPiecewiseLinearSeasonalArtifact(request) {
    const ret = wasm.fitPiecewiseLinearSeasonalArtifact(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {string} manifest_json
 * @returns {string}
 */
export function geoSplitManifestHash(manifest_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(manifest_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.geoSplitManifestHash(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * @param {string} artifact
 * @param {number} horizon
 * @returns {any}
 */
export function predictPiecewiseLinearSeasonalArtifact(artifact, horizon) {
    const ptr0 = passStringToWasm0(artifact, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.predictPiecewiseLinearSeasonalArtifact(ptr0, len0, horizon);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {string} artifact
 * @param {number} horizon
 * @param {any} options
 * @returns {any}
 */
export function predictPiecewiseLinearSeasonalArtifactWithOptions(artifact, horizon, options) {
    const ptr0 = passStringToWasm0(artifact, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.predictPiecewiseLinearSeasonalArtifactWithOptions(ptr0, len0, horizon, options);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runForecast(request) {
    const ret = wasm.runForecast(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runGeoCausalExperiment(request) {
    const ret = wasm.runGeoCausalExperiment(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runGeoFeatureExamples(request) {
    const ret = wasm.runGeoFeatureExamples(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runGeostatisticsModel(request) {
    const ret = wasm.runGeostatisticsModel(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runGeotemporalDiagnostics(request) {
    const ret = wasm.runGeotemporalDiagnostics(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runGraphForecast(request) {
    const ret = wasm.runGraphForecast(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runNeuralModel(request) {
    const ret = wasm.runNeuralModel(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runRegressionModel(request) {
    const ret = wasm.runRegressionModel(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} request
 * @returns {any}
 */
export function runSequence(request) {
    const ret = wasm.runSequence(request);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {any} features
 * @param {Float32Array} weights
 * @param {Float32Array} biases
 * @returns {Promise<any>}
 */
export function webgpuDenseLayer(features, weights, biases) {
    const ptr0 = passArrayF32ToWasm0(weights, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(biases, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.webgpuDenseLayer(features, ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Executes a browser WebGPU compute pass and resolves once the mapped output
 * has been verified. Unlike the synchronous modeling exports, this function
 * can await browser adapter and buffer-map promises safely.
 * @param {number} len
 * @returns {Promise<any>}
 */
export function webgpuDispatchReport(len) {
    const ret = wasm.webgpuDispatchReport(len);
    return ret;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_fdd633d4bb5dd76a: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_Number_c4bdf66bb78f7977: function(arg0) {
            const ret = Number(arg0);
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_Window_65ef42d29dc8174d: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_d272430d4a323303: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_bigint_get_as_i64_d9e915702856f831: function(arg0, arg1) {
            const v = arg1;
            const ret = typeof(v) === 'bigint' ? v : undefined;
            getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_boolean_get_edaed31a367ce1bd: function(arg0) {
            const v = arg0;
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_debug_string_8a447059637473e2: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_in_4990f46af709e33c: function(arg0, arg1) {
            const ret = arg0 in arg1;
            return ret;
        },
        __wbg___wbindgen_is_bigint_90b5ccfe67c78460: function(arg0) {
            const ret = typeof(arg0) === 'bigint';
            return ret;
        },
        __wbg___wbindgen_is_function_acc5528be2b923f2: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_6d937fbfb6478470: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_object_0beba4a1980d3eea: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_1fca8072260dd261: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_721f8decd50c87a3: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_jsval_eq_4e8c38722cb8ff51: function(arg0, arg1) {
            const ret = arg0 === arg1;
            return ret;
        },
        __wbg___wbindgen_jsval_loose_eq_4b9aba9e5b3c4582: function(arg0, arg1) {
            const ret = arg0 == arg1;
            return ret;
        },
        __wbg___wbindgen_number_get_1cc01dd708740256: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_71bb4348194e31f0: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_ea4887a5f8f9a9db: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_33c39e13d73b25f6: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_43b0c6751d870fcf: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_call_5575218572ead796: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_call_8e98ed2f3c86c4b5: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.call(arg1);
            return ret;
        }, arguments); },
        __wbg_copyBufferToBuffer_3b119149df2dc5eb: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4);
        }, arguments); },
        __wbg_copyBufferToBuffer_9e5aea97d7828aa3: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        }, arguments); },
        __wbg_createBindGroupLayout_59891d473ac8665d: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_4cb86ff853df5c69: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_3fa0256cba655273: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_98e3b731629054b4: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_9d101515d504e110: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_270b4fd0b4230373: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createShaderModule_f0aa469466c7bdaa: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_dispatchWorkgroups_26f6198195c36ca4: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_done_b62d4a7d2286852a: function(arg0) {
            const ret = arg0.done;
            return ret;
        },
        __wbg_end_8437a975bbfe0297: function(arg0) {
            arg0.end();
        },
        __wbg_entries_c261c3fa1f281256: function(arg0) {
            const ret = Object.entries(arg0);
            return ret;
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_finish_6c7bba424ffe1bbc: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_finish_c40b67ff2af88e0c: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_getMappedRange_59829576da3edd39: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_get_197a3fe98f169e38: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_9a29be2cb383ed9a: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_get_unchecked_54a4374c38e08460: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_with_ref_key_6412cf3094599694: function(arg0, arg1) {
            const ret = arg0[arg1];
            return ret;
        },
        __wbg_gpu_cbd27ad0589bc0b3: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_instanceof_ArrayBuffer_2a7bb09fee70c2da: function(arg0) {
            let result;
            try {
                result = arg0 instanceof ArrayBuffer;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuAdapter_1297a3a5ce0db3ff: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Map_afa18d5840c04c15: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Map;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Uint8Array_f080092dc70f5d58: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Uint8Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_isArray_145a34fd0a38d37b: function(arg0) {
            const ret = Array.isArray(arg0);
            return ret;
        },
        __wbg_isSafeInteger_a3389a198582f5f6: function(arg0) {
            const ret = Number.isSafeInteger(arg0);
            return ret;
        },
        __wbg_iterator_cc47ba25a2be735a: function() {
            const ret = Symbol.iterator;
            return ret;
        },
        __wbg_label_9a8583e3a20fafc7: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_589238bdcf171f0e: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_c6054974c0a6cdb9: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_mapAsync_e3cfbd141919d03c: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_navigator_017bc45e84c473cc: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_935098efd1dc7fe5: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_2e117a478906f062: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_3444eb7412549f0b: function() {
            const ret = new Map();
            return ret;
        },
        __wbg_new_36e147a8ced3c6e0: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_81880fb5002cb255: function(arg0) {
            const ret = new Uint8Array(arg0);
            return ret;
        },
        __wbg_new_typed_00a409eb4ec4f2d9: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen__convert__closures_____invoke__h39e1bf8ff3fdbdb9(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_f2b65504a914f37a: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_next_0c4066e251d2eff9: function() { return handleError(function (arg0) {
            const ret = arg0.next();
            return ret;
        }, arguments); },
        __wbg_next_402fa10b59ab20c3: function(arg0) {
            const ret = arg0.next;
            return ret;
        },
        __wbg_onSubmittedWorkDone_5f36409816d68e04: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_parse_1f9d3f9cbc8a7da2: function() { return handleError(function (arg0, arg1) {
            const ret = JSON.parse(getStringFromWasm0(arg0, arg1));
            return ret;
        }, arguments); },
        __wbg_prototypesetcall_d721637c7ca66eb8: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_f724b5db8acf89d2: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queueMicrotask_1c9b3800e321a967: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_311744e534a929a3: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_7bbf92178b06da19: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_0049683abd339828: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_921f0a221b4492fa: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_d82363d90af6928a: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_0500d49bcf971ad6: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setBindGroup_863d2daeb3c4fa01: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_c6aca1c13ec27120: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_272b80acaf9a75e8: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_4564f7dc44fcb0c9: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_9a1d61e17de7054c: function(arg0, arg1, arg2) {
            const ret = arg0.set(arg1, arg2);
            return ret;
        },
        __wbg_set_access_08d6bdbda9aaa266: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_beginning_of_pass_write_index_ebe753eeeade6f6c: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_078241cf2822c39e: function(arg0, arg1) {
            arg0.bindGroupLayouts = arg1;
        },
        __wbg_set_binding_d683cd9c1d4bcfed: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_e9ba14423117de0a: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_buffer_598ab98a251b8f91: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_73d9f6fea9c41867: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_code_6a0d763da082dcfb: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compute_5dd7704ee8a825c6: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_dc601f4a69da0bc2: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_end_of_pass_write_index_49de5f6017fb9a1f: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_070b048e4bea0c29: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entries_f9b7f3d4e9faccf4: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entry_point_52a2481a52f9799d: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_cf122b1392d58f37: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_format_27c63de9b0ec1cb3: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_has_dynamic_offset_69725fed837748fe: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_label_2a41a6f671383447: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_37d0faa0c9b7dee4: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_3e306b2e8f9db666: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_58fbc9fcc6363f16: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5a4dbb42c3b27bf7: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5c952448f9d59f36: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5fadf65a1f0f4714: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_782e33de78d86641: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_837a3b8ff99c2db3: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_8df6673e1e141fcc: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_cd5d951ba305620a: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_d701bf37a1e489c6: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_mapped_at_creation_7f0aad21612f3e22: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_min_binding_size_d70e460d165d9144: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_module_22d452288cef846d: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisampled_4ce4c32144215354: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_e316586bb85f0bd6: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_power_preference_7d669fb9b41f7bf2: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_query_set_604a8ae10429942b: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_required_features_3d00070d09235d7d: function(arg0, arg1) {
            arg0.requiredFeatures = arg1;
        },
        __wbg_set_required_limits_e0de55a49a48e3dc: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_fe1f979fce4afee2: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_sample_type_3cecbd4699e2e5fb: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_12544c21977075c1: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_size_0c20f73abce8f1ce: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_f1207de283144c72: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_storage_texture_36be4834c501acab: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_texture_738e6f6215515de3: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_6854d9d17bf5b0b4: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_17a1387b620bc902: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_type_d4edb621ec2051e0: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_usage_41b7d18f3f220e6c: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_dimension_4a840560a13b4860: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_9ae69db849267b1a: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_visibility_bbbf3d2b70571950: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_THIS_2fee5048bcca5938: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_ce44e66a4935da8c: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_44f6e0cb5e67cdad: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_168f178805d978fe: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_b3bbead76cbf7627: function(arg0, arg1) {
            arg0.submit(arg1);
        },
        __wbg_then_05edfc8a4fea5106: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_2a84678a50976959: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_591b6b3a75ee817a: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unmap_817a2e3248a553fb: function(arg0) {
            arg0.unmap();
        },
        __wbg_value_49f783bb59765962: function(arg0) {
            const ret = arg0.value;
            return ret;
        },
        __wbg_writeBuffer_24a10bfd5a8a57f7: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 489, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h0187be81d5918678);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 920, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h164c0d2c76578433);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0) {
            // Cast intrinsic for `I64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000007: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./cartoboost_wasm_bg.js": import0,
    };
}

function wasm_bindgen__convert__closures_____invoke__h0187be81d5918678(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h0187be81d5918678(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h164c0d2c76578433(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__h164c0d2c76578433(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h39e1bf8ff3fdbdb9(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen__convert__closures_____invoke__h39e1bf8ff3fdbdb9(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuBufferBindingType = ["uniform", "storage", "read-only-storage"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuSamplerBindingType = ["filtering", "non-filtering", "comparison"];


const __wbindgen_enum_GpuStorageTextureAccess = ["write-only", "read-only", "read-write"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureSampleType = ["float", "unfilterable-float", "depth", "sint", "uint"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('cartoboost_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
