# kineto-spyre 2.12.0 release

- **Release tag:** `torch-2.12.0.aiu.kineto.1.2.0` (git tag / GitHub release name; dotted convention)
- **PyTorch wheel version:** `torch-2.12.0+aiu.kineto.1.2.0` (PEP 440 `+` local version)
- **PyTorch target:** 2.12.0 (cp312)
- **Pinned upstream kineto commit:** `b2103f78d13fde4937af010c0ef8e24313568bc5`
- **Previous last sync commit:** `7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7`
- **Integrated upstream commits:** 85

> Subcomponent versions `libaiupti` and `aiu_toolkit` in `release_record.json` are placeholders (`TBD`) — set them to the actual installed versions before the release build (the build's subcomponent gate will otherwise stop).

Full provenance is in the committed `release_record.json`.

## Known issues / release blockers

### AIU trace generation requires a torch-2.12-compatible `torch_sendnn` (BLOCKER)

The trace-generation step (`scripts/gen_trace.py` → `tools/trace_validator`,
Req 3.3 / 7.1 / 7.2) cannot produce an AIU trace until a `torch_sendnn` build
that supports PyTorch 2.12 is available. The kineto-spyre wheel builds and the
profiler/aiupti plumbing is correct; the gap is entirely in the `torch_sendnn`
backend:

- **No eager AIU device.** `torch_sendnn` registers no eager PrivateUse1 device
  module (`torch._C._get_privateuse1_backend_name()` → `privateuseone`, default;
  `torch.randn(device="privateuseone")` raises
  `ModuleNotFoundError: No module named 'torch.privateuseone'`). AIU work must go
  through `torch.compile`.
- **`sendnn` compile backend is incompatible with torch 2.12.**
  `torch.compile(fn, backend="sendnn")` fails during AOTAutograd with
  `AttributeError: 'ViewAndMutationMeta' object has no attribute 'is_train'`
  (`torch_sendnn/backends/sendnn_backend.py:76`). The `is_train` field on
  AOTAutograd's `ViewAndMutationMeta` was removed/renamed in PyTorch 2.12, so the
  backend cannot compile.
- **The only versioned `torch_sendnn` pins old torch.** `torch_sendnn 1.2.2+0`
  requires `torch<=2.10.0,>=2.5.1`, incompatible with the
  `torch-2.12.0+aiu.kineto.1.2.0` wheel. The locally built `torch_sendnn 0.0.0`
  is the only torch-2.12-installable build, and it hits the `is_train` failure
  above.

**Resolution needed:** a `torch_sendnn` release updated for the PyTorch 2.12
AOTAutograd API (and, ideally, native PrivateUse1 profiler registration). Until
then, `gen_trace.py` can only be exercised end-to-end against AIU hardware with a
compatible backend; `AIU_COMPILE_BACKEND=sendnn_mock` can confirm the
profiler/validator plumbing but does not produce real AIU hardware events.

`gen_trace.py` already accommodates a future fixed backend: it auto-detects an
eager PrivateUse1 device, honors `AIU_DEVICE_NAME`, falls back to
`torch.compile` with `AIU_COMPILE_BACKEND` (default `sendnn`), and engages the
aiupti `ProfilerActivity=PrivateUse1` env-var fallback when the wheel lacks
native PrivateUse1 profiler registration.

## API changes (from the upstream kineto sync)

These public-header changes come from the integrated upstream commits
(`libkineto/include/`, range `7a731b6..b2103f78`). The AIU plugin's own API
(`AiuptiActivityApi`) is unchanged — the sync preserves it.

### Semantic changes

- **`ActivityType.h`**
  - Every enumerator now has an explicit numeric value (`USER_ANNOTATION = 1` … `ENUM_COUNT = 26`).
  - New activity types: `MTIA_COUNTERS = 17`, **`PRIVATEUSE1_RUNTIME = 24`**, **`PRIVATEUSE1_DRIVER = 25`** (PrivateUse1 = AIU-relevant).
  - `toString(ActivityType)` is now an inline, compile-time header function (no libkineto link needed); new `toActivityType(const std::string&)`; enum↔string map moved into the header; new `constexpr int defaultActivityTypeCount`.
- **`IActivityProfiler.h`**
  - `availableActivities()` is now `const` and pure virtual (`= 0`).
  - `processTrace(...)` and `configure(...)` signatures changed (e.g. `configure(const std::set<ActivityType>& activity_types, …)`).
  - `DeviceInfo` / `ResourceInfo` constructors take args by value + `std::move`; `ResourceInfo` argument order changed to `(deviceId, id, sortIndex, name)`.
- **`GenericTraceActivity.h`** — new public `addCounterValue(const std::string&, double)` and `counterValues()` (override), supporting counter events.
- **`ILoggerObserver.h`** — `LoggerOutputType` now has explicit values plus a new `USDT = 5` output type (`ENUM_COUNT = 6`); `kEmptyTrace` is `constexpr char[]`.
- **`EnvMetadata.h`** — adds `host_name` to trace metadata via `gethostname()` (+ Windows `ws2_32` link pragma).
- **`ActivityProfilerInterface.h`** — `addMetadata(...)` is pure virtual (`= 0`).

Most likely to affect downstream code: the `IActivityProfiler` signature/const
changes (anything implementing that interface) and the new `PRIVATEUSE1_*` /
`MTIA_COUNTERS` activity types.

### Mostly cosmetic (no behavioral/ABI impact)

clang-format reflows (multi-line signatures collapsed) and `[[maybe_unused]]`
parameter annotations across `ActivityProfilerInterface.h`,
`IActivityProfiler.h`, `libkineto.h`, `Config.h`, `AbstractConfig.h`,
`output_base.h`, `ITraceActivity.h`, `TraceSpan.h`, `time_since_epoch.h`.

> Derived from the header diff between the two pinned commits; the exact set is
> finalized when the cherry-pick sync runs.

## libkineto changes (PyTorch 2.11 -> 2.12)

The fork's bundled libkineto was synced from the 2.11-era pin
(`7a731b6`) to the exact commit **PyTorch 2.12** pins (`b2103f78`) — a delta of
**133 files in `libkineto/`, +8,036 / -4,784**. Why it was required: PyTorch's
`torch/csrc/profiler/kineto_shim.cpp` is compiled against the kineto API at the
commit that PyTorch version pins. PyTorch 2.12 references symbols that only
exist at `b2103f78` (e.g. `ActivityType::MTIA_COUNTERS`), so a fork still at the
2.11-era kineto fails to compile against the 2.12 source. There is no
kineto<->PyTorch compatibility guarantee, hence the exact-commit pin.

### Structural changes
- **Backend-agnostic profiler refactor:** common logic extracted from
  `CuptiActivityProfiler.cpp` (shrank ~1,485 lines) into the new
  `GenericActivityProfiler.{cpp,h}` (~1,550 lines) (#1219), so the
  CUPTI/ROCm/XPU/AIU backends plug into a shared structure.
- **ROCm moved to rocprofiler-sdk:** new `RocprofActivity*`, `RocprofLogger`,
  `RocLogger`, `RocmActivityProfiler` files; ROCm 6.4+ uses rocprofiler-sdk
  instead of libroctracer (#1249) (the CMake block merged next to the AIU guard).
- **CUPTI:** new `CuptiCbidRegistry.{cpp,h}`; `CuptiActivity` logic moved to
  headers; NCCL/GPU-kernel metadata and buffer-ordering fixes.
- **XPU:** profiler split into `XpuptiActivityProfilerSession`.
- **Chrome-trace writer** (`output_json.cpp`) substantially reworked.

### Public API / header changes (relevant to integrators)
- `ActivityType.h` — explicit enum values; new `MTIA_COUNTERS=17`,
  `PRIVATEUSE1_RUNTIME=24`, `PRIVATEUSE1_DRIVER=25`; inline `toString()` + new
  `toActivityType()`.
- `IActivityProfiler.h` — `availableActivities()` now `const`+pure-virtual;
  `processTrace()`/`configure()` signatures changed; `DeviceInfo`/`ResourceInfo`
  constructors changed.
- `GenericTraceActivity.h` — new `addCounterValue()` / `counterValues()`.
- `ILoggerObserver.h` — new `USDT` logger output type.
- `EnvMetadata.h` — adds `host_name` to trace metadata.
- `DeviceProperties` — per-backend `devicePropertiesJson()`/`smCount()`
  consolidated into single functions with internal `#if/#elif`.

### Impact on the AIU plugin (fork code)
The sync preserved AIU behavior, but two follow-on fixes were needed for 2.12:
- Re-insert the `HAS_AIUPTI` branch into the refactored consolidated
  `devicePropertiesJson()`.
- PyTorch 2.12 builds kineto with `-Wall -Wextra -pedantic -Werror`; latent
  AIU-plugin warnings (unused params, sign-compare, narrowing) became fatal and
  were fixed with `[[maybe_unused]]` and value-preserving casts.

## Integrated upstream kineto commits (85)



Range `7a731b6ae01c..b2103f78d13f` (first exclusive, last inclusive), ascending:

- `97be5d339ef0` Fill device properties for XPU (#1185)
- `2e15783f4cf9` Fix CQS signal modernize-pass-by-value in fbcode/kineto/libkineto
- `bee4c2dd95e7` Fix CQS signal readability-enum-initial-value in fbcode/kineto/libkineto
- `47093804abb9` Add basic Rocm CI build and test (#1231)
- `a087858d4463` update clang-tidy to reflect current code base (#1239)
- `445bc4f53d96` refactor activity profiler (#1219)
- `639a4106ef9c` Fix ROCm CI workflow (#1240)
- `c09ff739ebc8` Refactor ifdef in DeviceProperties (#1246)
- `8c171ba271e6` Refactor CUDA workflow to better use test-infra (#1244)
- `4b3b2ed2a6a1` Add deprecation warning for tb_plugin submodule (#1254)
- `81213d1fa59a` Return ResourceInfo with sycl_queue_id by xpu profiler (#1210)
- `62593b7e1f21` Refactor CPU and CUDA workflows (#1258)
- `f93df2c6640c` Introduce callback registry for callback management
- `21cbd296e4ca` Refactor CuptiActivity.cpp to use CallBackRegistry
- `d8ea052df62c` Re-enable unit tests (#1262)
- `ab377b922b67` Fix CQS signal readability-avoid-return-with-void-value in fbcode/kineto/libkineto
- `bf29d5ee7fab` Refactor CI ROCm workflow (#1260)
- `6a032a802088` Handle cuMem_ driver trace activities (#1263)
- `6d740c3dd5c1` add commas in clang-tidy (#1267)
- `aba03d0d7f79` Clear GPU activities during warmup for iteration-based profiling (#1268)
- `d8edadee854b` ensure architecture isolation through build system (#1261)
- `587a6ec49c00` Fix CQS signal readability-isolate-declaration in fbcode/kineto/libkineto
- `e8d2d27d841d` 2nd code cleanup of XPU profiler for incoming scope profiler (#1256)
- `0f9bef9a7caf` pull misnamed .cpp file into .h (#1270)
- `5dfd2b64b4f5` Fix CQS signal performance-inefficient-vector-operation in fbcode/kineto/libkineto
- `41bfe3fe65c4` Fix CQS signal facebook-unused-include-check in fbcode/kineto/libkineto [B]
- `7023dae77124` Fix ROCm test builds and enable activity profiler tests (#1271)
- `553beb12649d` Update .clang-format to align with PyTorch (#1272)
- `e3c937011ddb` Move RoctracerActivity.h includes to cpp file (#1274)
- `39afa0661717` Daily `arc lint --take CLANGFORMAT`
- `1a72dcae651d` Add graphId and graphNodeId to kernel metadata (#1276)
- `854891f776dc` Fix CQS signal modernize-use-designated-initializers in fbcode/kineto/libkineto
- `a2651fa6e505` Add graphId and graphNodeId to memset/memcopy nodes (#1277)
- `dadc7be773ab` rocprofiler-sdk support (#1249)
- `d5562f5d47fc` Fix CQS signal modernize-deprecated-headers in fbcode/kineto/libkineto
- `8d51d36b74d3` Fix CQS signal modernize-use-emplace in fbcode/kineto/libkineto
- `520b0564cdcb` Stream track descriptors before events in PerfettoTraceBuilder
- `f8881ce3f659` Fix flow event creation (#1229)
- `cf3cf5d8a55c` Fix linting CI workflow (#1287)
- `18154e907ffb` Enable pedantic compilation, fixup code that doesn't pass (#1282)
- `00355051f09e` Disable strict compiles for systems not in our CI (#1291)
- `a7c5f4d8f2c4` Add NCCL collective sequence number (seq_num) to Kineto profiler traces (#1294)
- `e2e7e97a1989` Revert D94566477: Add NCCL collective sequence number (seq_num) to Kineto profiler traces
- `ebaac17e387b` Add USDT log type to logger framework (#1285)
- `1f9ceb1289de` Use HAS_CUPTI_RANGE_PROFILER to avoid range profiler init (#1298)
- `350b58f0d6a2` Refactor CuptiActivityProfiler.cpp to use CuptiCbidRegistry (#1297)
- `2b15a605651b` Add seq_num propagation to GPU kernel events in Kineto trace output (#1296)
- `03ab8cb08c1b` Update ROCPROFILER_CALLBACK_* references to ROCPROFILER_BUFFER_* (#1295)
- `3b5cdcac8091` Add comms Id to trace output JSON (#1300)
- `c12ddc2d2fb7` refactor CuptiCbidRegistry member function names (#1301)
- `058386fcae20` Add Mac CPU workflow (#1304)
- `7d860f2786ee` Fix unit test (#1305)
- `e8956c4d7bf9` Integrate PyTorch's disabled tests mechanism into CI (#1311)
- `041c3e14cc16` Disable test_record_function_fast (#1309)
- `bb1e1942ceee` Add additionalLoggerCollector mechanism to ActivityProfilerController (#1290)
- `c6c84d098f0b` Use whole data from PTI activity record (#1278)
- `f882254e5ec1` Add MTIA_COUNTERS ActivityType and counter event output support (#1303)
- `0c52fa62fe31` Fix compilation of XPU part of kineto (#1292)
- `699f0920bb0a` Start flows on hipGraphLaunch (#1310)
- `2690e2096459` Add <chrono> header missing for Win build (#1312)
- `088abd294cac` Disable test_schedule_function_count which is segfaulting (#1320)
- `ad96ab49ea4b` Update .gitignore (#1315)
- `22cb9f15e528` Move activity_type enum <> string map to .h file (#1317)
- `8b42d4c1eb38` Add PYTORCH_TEST_WITH_ROCM=1 to test_profiler test for CI (#1289)
- `e19dd9266a14` Expose occupany limiting factors (#1322)
- `50a0085df709` Re-enabled some hardcoded tests (#1321)
- `3a616572a92c` Fix Lingering INT32 Overflow (#1324)
- `9d7373bfd148` Revert D97166802 (#1326)
- `628e1d0f967c` Add host_name to OSS Kineto trace metadata via gethostname() (#1323)
- `37fada9a1677` Ensure that async doesn't loop while sync is active (#1327)
- `4826a43a0729` Remove duplicate test ignore (#1328)
- `0c8ede0502ca` remove the rocprofiler early exit hack (#1329)
- `185fe9c531af` Expose occupany limiting factors (#1330)
- `502c5136f587` Remove INFO logs from kineto trace metadata to fix invalid JSON (#1336)
- `e335a7db4f1c` Remove tb_plugin CI workflows (#1341)
- `4ff89e88607e` Fix GPU trace collection on ROCm by ensuring early rocprofiler-sdk tool registration (#1338)
- `c51fcfeac5d7` refactor JSON chrome trace writing (#1342)
- `404f8cde73d6` Fix ABBA deadlock in bufferCompleted when verbose logging is enabled (#1347)
- `ec4a9b0aadee` approximate clock fast path for arm (#1333)
- `f689d8ad8f85` Fix missing wait_on_cuda_event_record_corr_id in Event Synchronize activities (#1318)
- `eea23344cad2` Fix XpuptiProfilerTest link error and double free crash (#1349)
- `f287ee74246f` Replace shared_ptr singleton with Meyers singleton in CuptiCallbackApi (#1350)
- `2b2647021545` Modernize target_link_libraries in Cmakelists.txt (#1283)
- `360873c004c3` Fix NCCL metadata missing on GPU kernels due to CUPTI buffer ordering (#1354)
- `b2103f78d13f` Emit USDT logger messages on profiler start/stop (#1325)

