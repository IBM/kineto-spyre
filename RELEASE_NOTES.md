# kineto-spyre 2.12.0 release

- **Release tag:** `torch-2.12.0.aiu.kineto.1.2.0` (git tag / GitHub release name; dotted convention)
- **PyTorch wheel version:** `torch-2.12.0+aiu.kineto.1.2.0` (PEP 440 `+` local version)
- **PyTorch target:** 2.12.0 (cp312)
- **Pinned upstream kineto commit:** `b2103f78d13fde4937af010c0ef8e24313568bc5`
- **Previous last sync commit:** `7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7`
- **Integrated upstream commits:** 85

> Subcomponent versions `libaiupti` and `aiu_toolkit` in `release_record.json` are placeholders (`TBD`) — set them to the actual installed versions before the release build (the build's subcomponent gate will otherwise stop).

Full provenance is in the committed `release_record.json`.

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

