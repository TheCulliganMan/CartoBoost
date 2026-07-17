//! HIP runtime backend.
//!
//! HIP does not currently have a cuda-oxide-equivalent native Rust toolchain.
//! Keep the vendor C ABI and HIPRTC source compilation isolated here so model
//! code remains safe Rust and non-HIP builds do not link ROCm libraries.

use crate::backend::{BackendDispatchReport, BackendSelection, CsrDiffusionBackward};
use crate::{AcceleratorError, Result};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString, OsString};
use std::path::{Path, PathBuf};
use std::time::Instant;

type HipError = i32;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HiprtcProgram = *mut c_void;

struct HipRuntime {
    _hip: libloading::Library,
    _rtc: libloading::Library,
    init: extern "C" fn(u32) -> HipError,
    get_device_count: extern "C" fn(*mut i32) -> HipError,
    set_device: extern "C" fn(i32) -> HipError,
    synchronize: extern "C" fn() -> HipError,
    malloc: extern "C" fn(*mut *mut c_void, usize) -> HipError,
    free: extern "C" fn(*mut c_void) -> HipError,
    memcpy_htod: extern "C" fn(*mut c_void, *const c_void, usize) -> HipError,
    memcpy_dtoh: extern "C" fn(*mut c_void, *const c_void, usize) -> HipError,
    module_load_data: extern "C" fn(*mut HipModule, *const c_void) -> HipError,
    module_unload: extern "C" fn(HipModule) -> HipError,
    module_get_function: extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError,
    module_launch_kernel: extern "C" fn(
        HipFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipError,
    rtc_create_program: extern "C" fn(
        *mut HiprtcProgram,
        *const c_char,
        *const c_char,
        i32,
        *const *const c_char,
        *const *const c_char,
    ) -> HipError,
    rtc_compile_program: extern "C" fn(HiprtcProgram, i32, *const *const c_char) -> HipError,
    rtc_get_code_size: extern "C" fn(HiprtcProgram, *mut usize) -> HipError,
    rtc_get_code: extern "C" fn(HiprtcProgram, *mut c_void) -> HipError,
    rtc_destroy_program: extern "C" fn(*mut HiprtcProgram) -> HipError,
    rtc_get_log_size: extern "C" fn(HiprtcProgram, *mut usize) -> HipError,
    rtc_get_log: extern "C" fn(HiprtcProgram, *mut c_char) -> HipError,
}

impl HipRuntime {
    fn new() -> Result<Self> {
        let hip = load_library(&hip_library_candidates(), "HIP runtime")?;
        let rtc = load_library(&hiprtc_library_candidates(), "HIPRTC")?;
        unsafe fn symbol<T: Copy>(library: &libloading::Library, name: &[u8]) -> Result<T> {
            library.get::<T>(name).map(|value| *value).map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to load HIP symbol {}: {error}",
                    String::from_utf8_lossy(name).trim_end_matches('\0')
                ))
            })
        }
        Ok(Self {
            init: unsafe { symbol(&hip, b"hipInit\0")? },
            get_device_count: unsafe { symbol(&hip, b"hipGetDeviceCount\0")? },
            set_device: unsafe { symbol(&hip, b"hipSetDevice\0")? },
            synchronize: unsafe { symbol(&hip, b"hipDeviceSynchronize\0")? },
            malloc: unsafe { symbol(&hip, b"hipMalloc\0")? },
            free: unsafe { symbol(&hip, b"hipFree\0")? },
            memcpy_htod: unsafe { symbol(&hip, b"hipMemcpyHtoD\0")? },
            memcpy_dtoh: unsafe { symbol(&hip, b"hipMemcpyDtoH\0")? },
            module_load_data: unsafe { symbol(&hip, b"hipModuleLoadData\0")? },
            module_unload: unsafe { symbol(&hip, b"hipModuleUnload\0")? },
            module_get_function: unsafe { symbol(&hip, b"hipModuleGetFunction\0")? },
            module_launch_kernel: unsafe { symbol(&hip, b"hipModuleLaunchKernel\0")? },
            rtc_create_program: unsafe { symbol(&rtc, b"hiprtcCreateProgram\0")? },
            rtc_compile_program: unsafe { symbol(&rtc, b"hiprtcCompileProgram\0")? },
            rtc_get_code_size: unsafe { symbol(&rtc, b"hiprtcGetCodeSize\0")? },
            rtc_get_code: unsafe { symbol(&rtc, b"hiprtcGetCode\0")? },
            rtc_destroy_program: unsafe { symbol(&rtc, b"hiprtcDestroyProgram\0")? },
            rtc_get_log_size: unsafe { symbol(&rtc, b"hiprtcGetProgramLogSize\0")? },
            rtc_get_log: unsafe { symbol(&rtc, b"hiprtcGetProgramLog\0")? },
            _hip: hip,
            _rtc: rtc,
        })
    }

    fn check(&self, status: HipError, context: &str) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(AcceleratorError::InvalidArgument(format!(
                "{context} (HIP error {status})"
            )))
        }
    }

    fn prepare(&self) -> Result<()> {
        self.check((self.init)(0), "failed to initialize HIP")?;
        let mut count = 0;
        self.check(
            (self.get_device_count)(&mut count),
            "failed to query HIP devices",
        )?;
        if count <= 0 {
            return Err(AcceleratorError::InvalidArgument(
                "no HIP device is available".to_string(),
            ));
        }
        let device = std::env::var("CARTOBOOST_HIP_DEVICE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0);
        if device < 0 || device >= count {
            return Err(AcceleratorError::InvalidArgument(format!(
                "CARTOBOOST_HIP_DEVICE={device} is outside the detected device range 0..{count}"
            )));
        }
        self.check((self.set_device)(device), "failed to select HIP device")
    }

    fn compile<T>(
        &self,
        source: &str,
        entry: &str,
        f: impl FnOnce(HipFunction) -> Result<T>,
    ) -> Result<T> {
        let source =
            CString::new(source).map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
        let name = CString::new("cartoboost_kernel.hip").expect("static kernel name");
        let entry =
            CString::new(entry).map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
        let mut program = std::ptr::null_mut();
        self.check(
            (self.rtc_create_program)(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ),
            "failed to create HIPRTC program",
        )?;
        let compile_status = (self.rtc_compile_program)(program, 0, std::ptr::null());
        if compile_status != 0 {
            let log = self.program_log(program);
            let _ = (self.rtc_destroy_program)(&mut program);
            return Err(AcceleratorError::InvalidArgument(format!(
                "failed to compile HIP kernel {} (HIPRTC error {}): {}",
                entry.to_string_lossy(),
                compile_status,
                log
            )));
        }
        let mut size = 0;
        self.check(
            (self.rtc_get_code_size)(program, &mut size),
            "failed to query HIP code size",
        )?;
        let mut code = vec![0_u8; size];
        self.check(
            (self.rtc_get_code)(program, code.as_mut_ptr().cast()),
            "failed to read HIP code",
        )?;
        self.check(
            (self.rtc_destroy_program)(&mut program),
            "failed to destroy HIPRTC program",
        )?;
        let mut module = std::ptr::null_mut();
        self.check(
            (self.module_load_data)(&mut module, code.as_ptr().cast()),
            "failed to load HIP module",
        )?;
        let mut function = std::ptr::null_mut();
        if let Err(error) = self.check(
            (self.module_get_function)(&mut function, module, entry.as_ptr()),
            "failed to find HIP kernel",
        ) {
            let _ = (self.module_unload)(module);
            return Err(error);
        }
        let result = f(function);
        let unload = self.check((self.module_unload)(module), "failed to unload HIP module");
        match (result, unload) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) | (Err(error), _) => Err(error),
        }
    }

    fn program_log(&self, program: HiprtcProgram) -> String {
        let mut size = 0;
        if (self.rtc_get_log_size)(program, &mut size) != 0 || size == 0 {
            return String::new();
        }
        let mut log = vec![0_u8; size];
        if (self.rtc_get_log)(program, log.as_mut_ptr().cast()) != 0 {
            return String::new();
        }
        let end = log.iter().position(|byte| *byte == 0).unwrap_or(log.len());
        String::from_utf8_lossy(&log[..end]).into_owned()
    }
}

thread_local! {
    static RUNTIME: RefCell<Option<HipRuntime>> = const { RefCell::new(None) };
}

fn with_runtime<T>(f: impl FnOnce(&HipRuntime) -> Result<T>) -> Result<T> {
    RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if runtime.is_none() {
            let value = HipRuntime::new()?;
            value.prepare()?;
            *runtime = Some(value);
        }
        f(runtime.as_ref().expect("initialized HIP runtime"))
    })
}

pub(crate) fn is_available() -> bool {
    with_runtime(|_| Ok(())).is_ok()
}

struct DeviceBuffer<'a> {
    runtime: &'a HipRuntime,
    ptr: *mut c_void,
}

impl<'a> DeviceBuffer<'a> {
    fn new(runtime: &'a HipRuntime, bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        runtime.check(
            (runtime.malloc)(&mut ptr, bytes.max(1)),
            "failed to allocate HIP memory",
        )?;
        Ok(Self { runtime, ptr })
    }

    fn upload<T: Copy>(runtime: &'a HipRuntime, values: &[T]) -> Result<Self> {
        let buffer = Self::new(runtime, std::mem::size_of_val(values))?;
        runtime.check(
            (runtime.memcpy_htod)(
                buffer.ptr,
                values.as_ptr().cast(),
                std::mem::size_of_val(values),
            ),
            "failed to upload HIP memory",
        )?;
        Ok(buffer)
    }

    fn download<T: Copy>(&self, output: &mut [T]) -> Result<()> {
        self.runtime.check(
            (self.runtime.memcpy_dtoh)(
                output.as_mut_ptr().cast(),
                self.ptr,
                std::mem::size_of_val(output),
            ),
            "failed to download HIP memory",
        )
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = (self.runtime.free)(self.ptr);
        }
    }
}

fn launch(
    runtime: &HipRuntime,
    function: HipFunction,
    count: usize,
    args: &mut [*mut c_void],
) -> Result<()> {
    let blocks = (count as u32).div_ceil(256);
    runtime.check(
        (runtime.module_launch_kernel)(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        ),
        "failed to launch HIP kernel",
    )?;
    runtime.check((runtime.synchronize)(), "failed to synchronize HIP kernel")
}

fn kernel_arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn library_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for key in ["HIP_PATH", "ROCM_PATH"] {
        if let Some(value) = std::env::var_os(key) {
            roots.push(PathBuf::from(value));
        }
    }
    #[cfg(target_os = "windows")]
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        let rocm = PathBuf::from(program_files).join("AMD").join("ROCm");
        if let Ok(entries) = std::fs::read_dir(rocm) {
            let mut versions = entries
                .flatten()
                .map(|entry| entry.path())
                .collect::<Vec<_>>();
            versions.sort_by(|a, b| b.cmp(a));
            roots.extend(versions);
        }
    }
    roots
}

fn candidates(names: &[&str]) -> Vec<OsString> {
    let mut values = Vec::new();
    for root in library_roots() {
        for name in names {
            values.push(root.join("bin").join(name).into_os_string());
        }
    }
    values.extend(names.iter().map(OsString::from));
    values
}

fn hip_library_candidates() -> Vec<OsString> {
    #[cfg(target_os = "windows")]
    {
        candidates(&["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"])
    }
    #[cfg(target_os = "linux")]
    {
        candidates(&["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"])
    }
}

fn hiprtc_library_candidates() -> Vec<OsString> {
    #[cfg(target_os = "windows")]
    {
        candidates(&[
            "hiprtc0700.dll",
            "hiprtc0604.dll",
            "hiprtc0603.dll",
            "hiprtc.dll",
        ])
    }
    #[cfg(target_os = "linux")]
    {
        candidates(&["libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"])
    }
}

fn load_library(candidates: &[OsString], label: &str) -> Result<libloading::Library> {
    for candidate in candidates {
        if let Ok(library) = unsafe { libloading::Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(AcceleratorError::InvalidArgument(format!(
        "failed to load {label}; searched {}",
        candidates
            .iter()
            .map(|v| Path::new(v).display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

const ELEMENTWISE_SOURCE: &str = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void affine(const float* x, const float* means, const float* weights, const float* intercepts, float* out, unsigned int rows, unsigned int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float value = intercepts[i];
    for (unsigned int col = 0; col < cols; ++col) value += (x[i * cols + col] - means[col]) * weights[col];
    out[i] = value;
}
extern "C" __global__ void dense(const float* x, const float* weights, const float* biases, float* out, unsigned int rows, unsigned int cols, unsigned int width) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    unsigned int row = i / width, output = i % width;
    float value = biases[output];
    for (unsigned int col = 0; col < cols; ++col) value += x[row * cols + col] * weights[col * width + output];
    out[i] = value;
}
extern "C" __global__ void pair_sigmoid(const float* embeddings, const unsigned int* pairs, float* out, unsigned int count, unsigned int width) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int left = pairs[2 * i], right = pairs[2 * i + 1];
    float dot = 0.0f;
    for (unsigned int j = 0; j < width; ++j) dot += embeddings[left * width + j] * embeddings[right * width + j];
    out[i] = 1.0f / (1.0f + expf(-dot));
}
extern "C" __global__ void csr_diffusion(const unsigned int* indptr, const unsigned int* indices, const float* weights, const float* values, float* output, unsigned int nodes, unsigned int channels, unsigned int total) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int channel = i % channels;
    unsigned int row = (i / channels) % nodes;
    unsigned int batch = i / (channels * nodes);
    float value = 0.0f;
    for (unsigned int edge = indptr[row]; edge < indptr[row + 1]; ++edge)
        value += weights[edge] * values[(batch * nodes + indices[edge]) * channels + channel];
    output[i] = value;
}
extern "C" __global__ void csr_input_grad(const unsigned int* indptr, const unsigned int* indices, const float* weights, const float* output_grad, float* input_grad, unsigned int nodes, unsigned int channels, unsigned int total) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int channel = i % channels;
    unsigned int row = (i / channels) % nodes;
    unsigned int batch = i / (channels * nodes);
    float gradient = output_grad[i];
    for (unsigned int edge = indptr[row]; edge < indptr[row + 1]; ++edge)
        atomicAdd(&input_grad[(batch * nodes + indices[edge]) * channels + channel], weights[edge] * gradient);
}
extern "C" __global__ void csr_edge_grad(const unsigned int* indptr, const unsigned int* indices, const float* values, const float* output_grad, float* edge_grad, unsigned int nodes, unsigned int channels, unsigned int batches, unsigned int edges) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edges) return;
    unsigned int row = 0;
    while (row + 1 < nodes && indptr[row + 1] <= edge) ++row;
    unsigned int source = indices[edge];
    float gradient = 0.0f;
    for (unsigned int batch = 0; batch < batches; ++batch)
        for (unsigned int channel = 0; channel < channels; ++channel)
            gradient += output_grad[(batch * nodes + row) * channels + channel] * values[(batch * nodes + source) * channels + channel];
    edge_grad[edge] = gradient;
}
extern "C" __global__ void csr_softmax(const unsigned int* indptr, const float* logits, float* output, unsigned int rows) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int begin = indptr[row], end = indptr[row + 1];
    if (begin == end) return;
    float maximum = logits[begin];
    for (unsigned int edge = begin + 1; edge < end; ++edge) maximum = fmaxf(maximum, logits[edge]);
    float denominator = 0.0f;
    for (unsigned int edge = begin; edge < end; ++edge) denominator += expf(logits[edge] - maximum);
    for (unsigned int edge = begin; edge < end; ++edge) output[edge] = expf(logits[edge] - maximum) / denominator;
}
extern "C" __global__ void csr_softmax_backward(const unsigned int* indptr, const float* weights, const float* output_grad, float* logits_grad, unsigned int rows) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int begin = indptr[row], end = indptr[row + 1];
    float dot = 0.0f;
    for (unsigned int edge = begin; edge < end; ++edge) dot += weights[edge] * output_grad[edge];
    for (unsigned int edge = begin; edge < end; ++edge) logits_grad[edge] = weights[edge] * (output_grad[edge] - dot);
}
extern "C" __global__ void adamw(float* parameters, float* first, float* second, const float* gradients, unsigned int count, float learning_rate, float weight_decay, float correction_first, float correction_second) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float gradient = gradients[i] + weight_decay * parameters[i];
    first[i] = 0.9f * first[i] + 0.1f * gradient;
    second[i] = 0.999f * second[i] + 0.001f * gradient * gradient;
    parameters[i] -= learning_rate * (first[i] / correction_first) / (sqrtf(second[i] / correction_second) + 1.0e-8f);
}
extern "C" __global__ void layer_norm(const float* values, const float* gamma, const float* beta, float* output, unsigned int rows, unsigned int width) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float mean = 0.0f;
    for (unsigned int col = 0; col < width; ++col) mean += values[row * width + col];
    mean /= (float)width;
    float variance = 0.0f;
    for (unsigned int col = 0; col < width; ++col) { float d = values[row * width + col] - mean; variance += d * d; }
    variance /= (float)width;
    float inverse = rsqrtf(variance + 1.0e-5f);
    for (unsigned int col = 0; col < width; ++col) output[row * width + col] = (values[row * width + col] - mean) * inverse * gamma[col] + beta[col];
}
extern "C" __global__ void scalar_graph(float* values, const unsigned char* opcodes, const unsigned int* left, const unsigned int* right, unsigned int count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (unsigned int i = 0; i < count; ++i) {
        unsigned char op = opcodes[i];
        if (op <= 1) continue;
        float a = values[left[i]], b = values[right[i]];
        if (op == 2) values[i] = a + b;
        else if (op == 3) values[i] = a * b;
        else if (op == 4) values[i] = a / fmaxf(b, 1.0e-12f);
        else if (op == 5) values[i] = tanhf(a);
        else if (op == 6) values[i] = expf(a);
        else if (op == 7) values[i] = sqrtf(fmaxf(a, 1.0e-12f));
        else if (op == 8) values[i] = sinf(a);
        else if (op == 9) values[i] = 1.0f / (1.0f + expf(-a));
        else if (op == 10) values[i] = fmaxf(a, b);
        else if (op == 11) values[i] = a;
    }
}
extern "C" __global__ void scalar_graph_train(float* values, const unsigned char* opcodes, const unsigned int* left, const unsigned int* right, const unsigned int* parameter_ids, float* parameters, float* first, float* second, float* gradients, float* parameter_gradients, unsigned int count, unsigned int loss, unsigned int parameter_count, unsigned int step, float learning_rate, float weight_decay) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for(unsigned int i=0;i<count;++i){unsigned char op=opcodes[i];if(op==1)values[i]=parameters[parameter_ids[i]];else if(op>1){float a=values[left[i]],b=values[right[i]];if(op==2)values[i]=a+b;else if(op==3)values[i]=a*b;else if(op==4)values[i]=a/fmaxf(b,1e-12f);else if(op==5)values[i]=tanhf(a);else if(op==6)values[i]=expf(a);else if(op==7)values[i]=sqrtf(fmaxf(a,1e-12f));else if(op==8)values[i]=sinf(a);else if(op==9)values[i]=1.0f/(1.0f+expf(-a));else if(op==10)values[i]=fmaxf(a,b);else if(op==11)values[i]=a;}}
    gradients[loss]=1.0f;
    for(unsigned int rev=count;rev>0;){--rev;unsigned char op=opcodes[rev];unsigned int l=left[rev],r=right[rev];float g=gradients[rev];if(op==1)parameter_gradients[parameter_ids[rev]]+=g;else if(op==2){gradients[l]+=g;gradients[r]+=g;}else if(op==3){gradients[l]+=g*values[r];gradients[r]+=g*values[l];}else if(op==4){float d=fmaxf(values[r],1e-12f);gradients[l]+=g/d;gradients[r]-=g*values[l]/(d*d);}else if(op==5)gradients[l]+=g*(1.0f-values[rev]*values[rev]);else if(op==6)gradients[l]+=g*values[rev];else if(op==7)gradients[l]+=g/(2.0f*fmaxf(values[rev],1e-12f));else if(op==8)gradients[l]+=g*cosf(values[l]);else if(op==9)gradients[l]+=g*values[rev]*(1.0f-values[rev]);else if(op==10)gradients[values[l]>=values[r]?l:r]+=g;else if(op==11)gradients[l]+=g;}
    float c1=1.0f-powf(0.9f,(float)step),c2=1.0f-powf(0.999f,(float)step);for(unsigned int i=0;i<parameter_count;++i){float g=parameter_gradients[i]+weight_decay*parameters[i];first[i]=0.9f*first[i]+0.1f*g;second[i]=0.999f*second[i]+0.001f*g*g;parameters[i]-=learning_rate*(first[i]/c1)/(sqrtf(second[i]/c2)+1e-8f);}
}
extern "C" __global__ void train_tanh_mlp(const float* inputs, const float* targets, float* p, unsigned int rows, unsigned int input_size, unsigned int hidden_size, unsigned int epochs, float learning_rate) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int b1 = hidden_size * input_size, w2 = b1 + hidden_size, b2 = w2 + hidden_size;
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) for (unsigned int row = 0; row < rows; ++row) {
        float prediction = p[b2];
        for (unsigned int h = 0; h < hidden_size; ++h) { float v = p[b1+h]; for (unsigned int i=0;i<input_size;++i) v += p[h*input_size+i]*inputs[row*input_size+i]; prediction += tanhf(v)*p[w2+h]; }
        float eg = 2.0f * (prediction-targets[row]); p[b2] -= learning_rate*eg;
        for (unsigned int h = 0; h < hidden_size; ++h) { float v=p[b1+h]; for(unsigned int i=0;i<input_size;++i) v += p[h*input_size+i]*inputs[row*input_size+i]; float a=tanhf(v), old=p[w2+h]; p[w2+h]-=learning_rate*eg*a; float g=eg*old*(1.0f-a*a); p[b1+h]-=learning_rate*g; for(unsigned int i=0;i<input_size;++i) p[h*input_size+i]-=learning_rate*g*inputs[row*input_size+i]; }
    }
}
"#;

pub(crate) fn vector_add_report(
    selection: BackendSelection,
    len: usize,
    expected_checksum: f64,
) -> Result<BackendDispatchReport> {
    let left = (0..len).map(|i| i as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|i| i as f32 * 1.5).collect::<Vec<_>>();
    let start = Instant::now();
    let output = with_runtime(|runtime| {
        let left = DeviceBuffer::upload(runtime, &left)?;
        let right = DeviceBuffer::upload(runtime, &right)?;
        let output = DeviceBuffer::new(runtime, len * 4)?;
        runtime.compile(ELEMENTWISE_SOURCE, "vector_add", |function| {
            let mut a = left.ptr;
            let mut b = right.ptr;
            let mut out = output.ptr;
            let mut n = len as u32;
            let mut args = [
                kernel_arg(&mut a),
                kernel_arg(&mut b),
                kernel_arg(&mut out),
                kernel_arg(&mut n),
            ];
            launch(runtime, function, len, &mut args)
        })?;
        let mut values = vec![0.0_f32; len];
        output.download(&mut values)?;
        Ok(values)
    })?;
    Ok(BackendDispatchReport {
        requested: selection.requested,
        selected: selection.selected,
        operation: "vector_add".to_string(),
        len,
        checksum: output.into_iter().map(f64::from).sum(),
        expected_checksum,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        accelerated: true,
    })
}

pub(crate) fn affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    biases: &[f64],
) -> Result<Vec<f64>> {
    let rows = features.len();
    let cols = means.len();
    let x = features
        .iter()
        .flatten()
        .map(|v| *v as f32)
        .collect::<Vec<_>>();
    let means = means.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let weights = weights.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let biases = biases.iter().map(|v| *v as f32).collect::<Vec<_>>();
    run_kernel_output("affine", rows, rows, |runtime, function, output| {
        let x = DeviceBuffer::upload(runtime, &x)?;
        let means = DeviceBuffer::upload(runtime, &means)?;
        let weights = DeviceBuffer::upload(runtime, &weights)?;
        let biases = DeviceBuffer::upload(runtime, &biases)?;
        let (mut xp, mut mp, mut wp, mut bp, mut op) =
            (x.ptr, means.ptr, weights.ptr, biases.ptr, output.ptr);
        let (mut rows, mut cols) = (rows as u32, cols as u32);
        let mut args = [
            kernel_arg(&mut xp),
            kernel_arg(&mut mp),
            kernel_arg(&mut wp),
            kernel_arg(&mut bp),
            kernel_arg(&mut op),
            kernel_arg(&mut rows),
            kernel_arg(&mut cols),
        ];
        launch(runtime, function, rows as usize, &mut args)
    })
    .map(|v| v.into_iter().map(f64::from).collect())
}

pub(crate) fn dense_layer(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let rows = features.len();
    let cols = features[0].len();
    let width = biases.len();
    let x = features.iter().flatten().copied().collect::<Vec<_>>();
    let output = run_matrix_kernel("dense", &x, None, weights, biases, rows, cols, width)?;
    Ok(output.chunks(width).map(<[f32]>::to_vec).collect())
}

fn run_matrix_kernel(
    entry: &str,
    x: &[f32],
    means: Option<&[f32]>,
    weights: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    width: usize,
) -> Result<Vec<f32>> {
    with_runtime(|runtime| {
        let x = DeviceBuffer::upload(runtime, x)?;
        let weights = DeviceBuffer::upload(runtime, weights)?;
        let biases = DeviceBuffer::upload(runtime, biases)?;
        let means_buffer = means
            .map(|v| DeviceBuffer::upload(runtime, v))
            .transpose()?;
        let output = DeviceBuffer::new(runtime, rows * width * 4)?;
        runtime.compile(ELEMENTWISE_SOURCE, entry, |function| {
            let mut x_ptr = x.ptr;
            let mut means_ptr = means_buffer
                .as_ref()
                .map_or(std::ptr::null_mut(), |v| v.ptr);
            let mut weights_ptr = weights.ptr;
            let mut biases_ptr = biases.ptr;
            let mut output_ptr = output.ptr;
            let mut rows = rows as u32;
            let mut cols = cols as u32;
            let mut width = width as u32;
            let mut args = if means.is_some() {
                vec![
                    kernel_arg(&mut x_ptr),
                    kernel_arg(&mut means_ptr),
                    kernel_arg(&mut weights_ptr),
                    kernel_arg(&mut biases_ptr),
                    kernel_arg(&mut output_ptr),
                    kernel_arg(&mut rows),
                    kernel_arg(&mut cols),
                    kernel_arg(&mut width),
                ]
            } else {
                vec![
                    kernel_arg(&mut x_ptr),
                    kernel_arg(&mut weights_ptr),
                    kernel_arg(&mut biases_ptr),
                    kernel_arg(&mut output_ptr),
                    kernel_arg(&mut rows),
                    kernel_arg(&mut cols),
                    kernel_arg(&mut width),
                ]
            };
            launch(runtime, function, rows as usize * width as usize, &mut args)
        })?;
        let mut values = vec![0.0; rows as usize * width as usize];
        output.download(&mut values)?;
        Ok(values)
    })
}

pub(crate) fn pair_sigmoid_scores(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    let width = embeddings[0].len();
    let flat = embeddings.iter().flatten().copied().collect::<Vec<_>>();
    let pairs = pairs
        .iter()
        .flat_map(|(a, b)| [*a as u32, *b as u32])
        .collect::<Vec<_>>();
    run_kernel_output(
        "pair_sigmoid",
        pairs.len() / 2,
        pairs.len() / 2,
        |runtime, function, output| {
            let e = DeviceBuffer::upload(runtime, &flat)?;
            let p = DeviceBuffer::upload(runtime, &pairs)?;
            let mut ep = e.ptr;
            let mut pp = p.ptr;
            let mut op = output.ptr;
            let mut count = (pairs.len() / 2) as u32;
            let mut width = width as u32;
            let mut args = [
                kernel_arg(&mut ep),
                kernel_arg(&mut pp),
                kernel_arg(&mut op),
                kernel_arg(&mut count),
                kernel_arg(&mut width),
            ];
            launch(runtime, function, count as usize, &mut args)
        },
    )
    .map(|v| v.into_iter().map(f64::from).collect())
}

pub(crate) fn csr_diffusion(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    let nodes = indptr.len() - 1;
    let total = values.len();
    run_kernel_output(
        "csr_diffusion",
        total,
        total,
        |runtime, function, output| {
            let ip = DeviceBuffer::upload(runtime, indptr)?;
            let ix = DeviceBuffer::upload(runtime, indices)?;
            let w = DeviceBuffer::upload(runtime, weights)?;
            let v = DeviceBuffer::upload(runtime, values)?;
            let (mut ipp, mut ixp, mut wp, mut vp, mut op) =
                (ip.ptr, ix.ptr, w.ptr, v.ptr, output.ptr);
            let (mut n, mut c, mut t) = (nodes as u32, channels as u32, total as u32);
            let mut args = [
                kernel_arg(&mut ipp),
                kernel_arg(&mut ixp),
                kernel_arg(&mut wp),
                kernel_arg(&mut vp),
                kernel_arg(&mut op),
                kernel_arg(&mut n),
                kernel_arg(&mut c),
                kernel_arg(&mut t),
            ];
            launch(runtime, function, total, &mut args)
        },
    )
}

pub(crate) fn csr_diffusion_backward(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    let nodes = indptr.len() - 1;
    let batches = values.len() / (nodes * channels);
    with_runtime(|runtime| {
        let ip = DeviceBuffer::upload(runtime, indptr)?;
        let ix = DeviceBuffer::upload(runtime, indices)?;
        let w = DeviceBuffer::upload(runtime, weights)?;
        let v = DeviceBuffer::upload(runtime, values)?;
        let og = DeviceBuffer::upload(runtime, output_grad)?;
        let ig = DeviceBuffer::upload(runtime, &vec![0.0_f32; values.len()])?;
        let eg = DeviceBuffer::new(runtime, weights.len() * 4)?;
        runtime.compile(ELEMENTWISE_SOURCE, "csr_input_grad", |f| {
            let (mut ipp, mut ixp, mut wp, mut ogp, mut igp) =
                (ip.ptr, ix.ptr, w.ptr, og.ptr, ig.ptr);
            let (mut n, mut c, mut t) = (nodes as u32, channels as u32, values.len() as u32);
            let mut a = [
                kernel_arg(&mut ipp),
                kernel_arg(&mut ixp),
                kernel_arg(&mut wp),
                kernel_arg(&mut ogp),
                kernel_arg(&mut igp),
                kernel_arg(&mut n),
                kernel_arg(&mut c),
                kernel_arg(&mut t),
            ];
            launch(runtime, f, values.len(), &mut a)
        })?;
        runtime.compile(ELEMENTWISE_SOURCE, "csr_edge_grad", |f| {
            let (mut ipp, mut ixp, mut vp, mut ogp, mut egp) =
                (ip.ptr, ix.ptr, v.ptr, og.ptr, eg.ptr);
            let (mut n, mut c, mut b, mut e) = (
                nodes as u32,
                channels as u32,
                batches as u32,
                weights.len() as u32,
            );
            let mut a = [
                kernel_arg(&mut ipp),
                kernel_arg(&mut ixp),
                kernel_arg(&mut vp),
                kernel_arg(&mut ogp),
                kernel_arg(&mut egp),
                kernel_arg(&mut n),
                kernel_arg(&mut c),
                kernel_arg(&mut b),
                kernel_arg(&mut e),
            ];
            launch(runtime, f, weights.len(), &mut a)
        })?;
        let mut input_grad = vec![0.0; values.len()];
        let mut edge_grad = vec![0.0; weights.len()];
        ig.download(&mut input_grad)?;
        eg.download(&mut edge_grad)?;
        Ok(CsrDiffusionBackward {
            input_grad,
            edge_grad,
        })
    })
}

pub(crate) fn csr_row_softmax(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    let rows = indptr.len() - 1;
    run_kernel_output("csr_softmax", rows, logits.len(), |runtime, f, out| {
        let ip = DeviceBuffer::upload(runtime, indptr)?;
        let l = DeviceBuffer::upload(runtime, logits)?;
        let (mut ipp, mut lp, mut op) = (ip.ptr, l.ptr, out.ptr);
        let mut r = rows as u32;
        let mut a = [
            kernel_arg(&mut ipp),
            kernel_arg(&mut lp),
            kernel_arg(&mut op),
            kernel_arg(&mut r),
        ];
        launch(runtime, f, rows, &mut a)
    })
}
pub(crate) fn csr_row_softmax_backward(
    indptr: &[u32],
    weights: &[f32],
    output_grad: &[f32],
) -> Result<Vec<f32>> {
    let rows = indptr.len() - 1;
    run_kernel_output(
        "csr_softmax_backward",
        rows,
        weights.len(),
        |runtime, f, out| {
            let ip = DeviceBuffer::upload(runtime, indptr)?;
            let w = DeviceBuffer::upload(runtime, weights)?;
            let g = DeviceBuffer::upload(runtime, output_grad)?;
            let (mut ipp, mut wp, mut gp, mut op) = (ip.ptr, w.ptr, g.ptr, out.ptr);
            let mut r = rows as u32;
            let mut a = [
                kernel_arg(&mut ipp),
                kernel_arg(&mut wp),
                kernel_arg(&mut gp),
                kernel_arg(&mut op),
                kernel_arg(&mut r),
            ];
            launch(runtime, f, rows, &mut a)
        },
    )
}

pub(crate) fn adamw(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    gradients: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    with_runtime(|runtime| {
        let p = DeviceBuffer::upload(runtime, parameters)?;
        let m = DeviceBuffer::upload(runtime, first)?;
        let v = DeviceBuffer::upload(runtime, second)?;
        let g = DeviceBuffer::upload(runtime, gradients)?;
        runtime.compile(ELEMENTWISE_SOURCE, "adamw", |f| {
            let (mut pp, mut mp, mut vp, mut gp) = (p.ptr, m.ptr, v.ptr, g.ptr);
            let mut n = parameters.len() as u32;
            let (mut lr, mut wd) = (learning_rate, weight_decay);
            let mut c1 = 1.0 - 0.9_f32.powi(step as i32);
            let mut c2 = 1.0 - 0.999_f32.powi(step as i32);
            let mut a = [
                kernel_arg(&mut pp),
                kernel_arg(&mut mp),
                kernel_arg(&mut vp),
                kernel_arg(&mut gp),
                kernel_arg(&mut n),
                kernel_arg(&mut lr),
                kernel_arg(&mut wd),
                kernel_arg(&mut c1),
                kernel_arg(&mut c2),
            ];
            launch(runtime, f, parameters.len(), &mut a)
        })?;
        p.download(parameters)?;
        m.download(first)?;
        v.download(second)
    })
}

pub(crate) fn layer_norm(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    run_kernel_output("layer_norm", rows, values.len(), |runtime, f, out| {
        let v = DeviceBuffer::upload(runtime, values)?;
        let g = DeviceBuffer::upload(runtime, gamma)?;
        let b = DeviceBuffer::upload(runtime, beta)?;
        let (mut vp, mut gp, mut bp, mut op) = (v.ptr, g.ptr, b.ptr, out.ptr);
        let (mut r, mut w) = (rows as u32, width as u32);
        let mut a = [
            kernel_arg(&mut vp),
            kernel_arg(&mut gp),
            kernel_arg(&mut bp),
            kernel_arg(&mut op),
            kernel_arg(&mut r),
            kernel_arg(&mut w),
        ];
        launch(runtime, f, rows, &mut a)
    })
}

pub(crate) fn scalar_graph(
    initial: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    with_runtime(|runtime| {
        let values = DeviceBuffer::upload(runtime, initial)?;
        let ops = DeviceBuffer::upload(runtime, opcodes)?;
        let l = DeviceBuffer::upload(runtime, left)?;
        let r = DeviceBuffer::upload(runtime, right)?;
        runtime.compile(ELEMENTWISE_SOURCE, "scalar_graph", |f| {
            let (mut vp, mut op, mut lp, mut rp) = (values.ptr, ops.ptr, l.ptr, r.ptr);
            let mut n = initial.len() as u32;
            let mut a = [
                kernel_arg(&mut vp),
                kernel_arg(&mut op),
                kernel_arg(&mut lp),
                kernel_arg(&mut rp),
                kernel_arg(&mut n),
            ];
            launch(runtime, f, 1, &mut a)
        })?;
        let mut output = vec![0.0; initial.len()];
        values.download(&mut output)?;
        Ok(output)
    })
}

pub(crate) fn train_tanh_mlp(
    inputs: &[Vec<f32>],
    targets: &[f32],
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    parameters: &mut [f32],
) -> Result<()> {
    let flat = inputs.iter().flatten().copied().collect::<Vec<_>>();
    with_runtime(|runtime| {
        let x = DeviceBuffer::upload(runtime, &flat)?;
        let y = DeviceBuffer::upload(runtime, targets)?;
        let p = DeviceBuffer::upload(runtime, parameters)?;
        runtime.compile(ELEMENTWISE_SOURCE, "train_tanh_mlp", |f| {
            let (mut xp, mut yp, mut pp) = (x.ptr, y.ptr, p.ptr);
            let mut rows = inputs.len() as u32;
            let mut width = inputs[0].len() as u32;
            let mut hidden = hidden_size as u32;
            let mut epochs = epochs as u32;
            let mut lr = learning_rate;
            let mut a = [
                kernel_arg(&mut xp),
                kernel_arg(&mut yp),
                kernel_arg(&mut pp),
                kernel_arg(&mut rows),
                kernel_arg(&mut width),
                kernel_arg(&mut hidden),
                kernel_arg(&mut epochs),
                kernel_arg(&mut lr),
            ];
            launch(runtime, f, 1, &mut a)
        })?;
        p.download(parameters)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_graph_train_step(
    initial: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
    parameter_ids: &[u32],
    loss: usize,
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<f32> {
    with_runtime(|runtime| {
        let values = DeviceBuffer::upload(runtime, initial)?;
        let ops = DeviceBuffer::upload(runtime, opcodes)?;
        let l = DeviceBuffer::upload(runtime, left)?;
        let r = DeviceBuffer::upload(runtime, right)?;
        let ids = DeviceBuffer::upload(runtime, parameter_ids)?;
        let p = DeviceBuffer::upload(runtime, parameters)?;
        let m = DeviceBuffer::upload(runtime, first)?;
        let v = DeviceBuffer::upload(runtime, second)?;
        let g = DeviceBuffer::upload(runtime, &vec![0.0f32; initial.len()])?;
        let pg = DeviceBuffer::upload(runtime, &vec![0.0f32; parameters.len()])?;
        runtime.compile(ELEMENTWISE_SOURCE,"scalar_graph_train",|f|{let(mut vp,mut op,mut lp,mut rp,mut ip,mut pp,mut mp,mut sp,mut gp,mut pgp)=(values.ptr,ops.ptr,l.ptr,r.ptr,ids.ptr,p.ptr,m.ptr,v.ptr,g.ptr,pg.ptr);let mut n=initial.len() as u32;let mut loss=loss as u32;let mut pn=parameters.len() as u32;let mut step=step as u32;let mut lr=learning_rate;let mut wd=weight_decay;let mut a=[kernel_arg(&mut vp),kernel_arg(&mut op),kernel_arg(&mut lp),kernel_arg(&mut rp),kernel_arg(&mut ip),kernel_arg(&mut pp),kernel_arg(&mut mp),kernel_arg(&mut sp),kernel_arg(&mut gp),kernel_arg(&mut pgp),kernel_arg(&mut n),kernel_arg(&mut loss),kernel_arg(&mut pn),kernel_arg(&mut step),kernel_arg(&mut lr),kernel_arg(&mut wd)];launch(runtime,f,1,&mut a)})?;
        let mut computed = vec![0.0; initial.len()];
        values.download(&mut computed)?;
        p.download(parameters)?;
        m.download(first)?;
        v.download(second)?;
        Ok(computed[loss])
    })
}

fn run_kernel_output(
    entry: &str,
    launch_count: usize,
    output_len: usize,
    body: impl FnOnce(&HipRuntime, HipFunction, &DeviceBuffer<'_>) -> Result<()>,
) -> Result<Vec<f32>> {
    with_runtime(|runtime| {
        let output = DeviceBuffer::new(runtime, output_len * 4)?;
        runtime.compile(ELEMENTWISE_SOURCE, entry, |function| {
            body(runtime, function, &output)
        })?;
        let mut values = vec![0.0; output_len];
        output.download(&mut values)?;
        let _ = launch_count;
        Ok(values)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        backend_adamw_step_f32, backend_csr_diffusion_backward_f32, backend_csr_diffusion_f32,
        backend_csr_row_softmax_backward_f32, backend_csr_row_softmax_f32, backend_layer_norm_f32,
        select_backend,
    };

    #[test]
    fn hip_graph_training_primitives_match_cpu_on_device() {
        if !is_available() {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let hip = select_backend(Some("rocm")).unwrap();
        let indptr = [0, 2, 3, 3];
        let indices = [0, 1, 2];
        let weights = [0.25, 0.75, -0.5];
        let values = [
            1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        ];
        let output_grad = [0.5; 12];
        let expected =
            backend_csr_diffusion_f32(&cpu, &indptr, &indices, &weights, 2, &values).unwrap();
        let actual =
            backend_csr_diffusion_f32(&hip, &indptr, &indices, &weights, 2, &values).unwrap();
        let expected_backward = backend_csr_diffusion_backward_f32(
            &cpu,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        let actual_backward = backend_csr_diffusion_backward_f32(
            &hip,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        for (expected, actual) in expected
            .iter()
            .zip(actual)
            .chain(
                expected_backward
                    .input_grad
                    .iter()
                    .zip(actual_backward.input_grad),
            )
            .chain(
                expected_backward
                    .edge_grad
                    .iter()
                    .zip(actual_backward.edge_grad),
            )
        {
            assert!((expected - actual).abs() < 1.0e-4, "{expected} != {actual}");
        }

        let logits = [1.0, -0.5, 2.0];
        let output_gradient = [0.5, -1.0, 0.25];
        let expected = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        let actual = backend_csr_row_softmax_f32(&hip, &indptr, &logits).unwrap();
        let expected_gradient =
            backend_csr_row_softmax_backward_f32(&cpu, &indptr, &expected, &output_gradient)
                .unwrap();
        let actual_gradient =
            backend_csr_row_softmax_backward_f32(&hip, &indptr, &actual, &output_gradient).unwrap();
        for (expected, actual) in expected
            .iter()
            .zip(actual)
            .chain(expected_gradient.iter().zip(actual_gradient))
        {
            assert!((expected - actual).abs() < 1.0e-5);
        }

        let values = [1.0, -2.0, 3.0, 4.0, -1.0, 0.5];
        let gamma = [1.0, 0.5, -0.25];
        let beta = [0.1, -0.2, 0.3];
        let expected = backend_layer_norm_f32(&cpu, &values, 2, 3, &gamma, &beta).unwrap();
        let actual = backend_layer_norm_f32(&hip, &values, 2, 3, &gamma, &beta).unwrap();
        for (expected, actual) in expected.iter().zip(actual) {
            assert!((expected - actual).abs() < 1.0e-5);
        }

        let mut expected_parameters = vec![1.0, -2.0, 0.5];
        let mut actual_parameters = expected_parameters.clone();
        let mut expected_first = vec![0.0; 3];
        let mut actual_first = expected_first.clone();
        let mut expected_second = vec![0.0; 3];
        let mut actual_second = expected_second.clone();
        backend_adamw_step_f32(
            &cpu,
            &mut expected_parameters,
            &mut expected_first,
            &mut expected_second,
            &[0.5, -0.25, 1.0],
            1,
            0.01,
            0.001,
        )
        .unwrap();
        backend_adamw_step_f32(
            &hip,
            &mut actual_parameters,
            &mut actual_first,
            &mut actual_second,
            &[0.5, -0.25, 1.0],
            1,
            0.01,
            0.001,
        )
        .unwrap();
        for (expected, actual) in expected_parameters
            .iter()
            .zip(actual_parameters)
            .chain(expected_first.iter().zip(actual_first))
            .chain(expected_second.iter().zip(actual_second))
        {
            assert!((expected - actual).abs() < 1.0e-6);
        }

        let graph = scalar_graph(
            &[2.0, 3.0, 0.0, 0.0, 0.0],
            &[0, 0, 2, 3, 5],
            &[0, 0, 0, 2, 3],
            &[0, 0, 1, 1, 0],
        )
        .unwrap();
        assert!((graph[4] - 15.0_f32.tanh()).abs() < 1.0e-6);
    }
}
