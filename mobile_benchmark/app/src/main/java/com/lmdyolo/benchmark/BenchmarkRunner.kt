package com.lmdyolo.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.EnumSet
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

enum class Backend(val wireName: String, val label: String) {
    XNNPACK("cpu_xnnpack", "CPU/XNNPACK"),
    NNAPI("nnapi", "NNAPI")
}

data class Detection(
    val classId: Int,
    val score: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

data class StageTimes(
    val decodeMs: Double,
    val resizeMs: Double,
    val tensorMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val nmsMs: Double,
    val totalMs: Double
) {
    val preprocessingMs: Double get() = decodeMs + resizeMs + tensorMs
}

data class PipelineResult(
    val times: StageTimes,
    val output: FloatArray,
    val candidates: Int,
    val detections: List<Detection>,
    val input: FloatArray? = null
)

class BenchmarkRunner(
    private val context: Context,
    private val reportLine: (String) -> Unit
) {
    companion object {
        const val TAG = "LMD_BENCH"
        private const val MODEL_ASSET = "best_deploy_fp32_source_copy.onnx"
        private const val IMAGE_ASSET = "China_Drone_000001.jpg"
        private const val GOLDEN_INPUT_ASSET = "golden_input_fp32.bin"
        private const val GOLDEN_OUTPUT_ASSET = "golden_output_fp32.bin"
        private const val GOLDEN_META_ASSET = "golden_metadata.json"
        private const val CANONICAL_SOURCE_RGB_ASSET = "canonical_source_rgb_u8.bin"
        private const val CANONICAL_INPUT_ASSET = "canonical_input_fp32.bin"
        private const val CANONICAL_OUTPUT_ASSET = "canonical_output_fp32.bin"
        private const val CANONICAL_META_ASSET = "canonical_preprocessing_metadata.json"
        private const val MODEL_SHA256 = "a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8"
        private const val SIZE = 640
        private const val CHANNELS = 3
        private const val OUTPUT_CHANNELS = 8
        private const val ANCHORS = 8400
        private const val WARMUPS = 10
        private const val MEASURED_RUNS = 100
        private const val CONFIDENCE = 0.25f
        private const val IOU = 0.45f
        private const val XNNPACK_THREADS = 4
        private val CLASS_NAMES = arrayOf("D00", "D10", "D20", "D40")

        fun cpuGateFile(context: Context): File = File(context.filesDir, "cpu_xnnpack_pass.json")
        fun nnapiValidationGateFile(context: Context): File = File(context.filesDir, "nnapi_validation_pass.json")
    }

    private val environment = OrtEnvironment.getEnvironment()

    /** One-shot CPU validation; it never starts the sustained benchmark or NNAPI. */
    fun runPreprocessingValidation(): JSONObject {
        val report = baseReport(Backend.XNNPACK).apply {
            put("phase", "canonical_preprocessing_validation")
            put("warmups_requested", 0)
            put("measured_runs_requested", 0)
            put("scope", "JPEG decode + canonical letterbox/RGB/NCHW + one ONNX inference")
        }
        val modelFile = File(context.noBackupFilesDir, MODEL_ASSET)
        context.assets.open(MODEL_ASSET).use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }
        val deviceHash = sha256(modelFile)
        report.put("model_sha256_device", deviceHash)
        check(deviceHash.equals(MODEL_SHA256, ignoreCase = true)) { "SHA-256 del ONNX no coincide" }

        val metadata = JSONObject(context.assets.open(CANONICAL_META_ASSET).bufferedReader().use { it.readText() })
        val sourceShape = metadata.getJSONArray("source_rgb_shape")
        val sourceHeight = sourceShape.getInt(0)
        val sourceWidth = sourceShape.getInt(1)
        val pcSourceRgb = context.assets.open(CANONICAL_SOURCE_RGB_ASSET).use { it.readBytes() }
        check(pcSourceRgb.size == sourceWidth * sourceHeight * 3)
        val goldenInput = readFloatAsset(CANONICAL_INPUT_ASSET, CHANNELS * SIZE * SIZE)
        val goldenOutput = readFloatAsset(CANONICAL_OUTPUT_ASSET, OUTPUT_CHANNELS * ANCHORS)
        val imageBytes = context.assets.open(IMAGE_ASSET).use { it.readBytes() }
        report.put("canonical_metadata", metadata)

        val options = sessionOptions(Backend.XNNPACK, profiling = false)
        val session = environment.createSession(modelFile.absolutePath, options)
        try {
            report.put("model_io", validateModelIo(session))

            // Exact test: both platforms receive the very same decoded RGB8 matrix.
            val pcRgbImage = RgbImage(sourceWidth, sourceHeight, pcSourceRgb)
            val exactTensor = CanonicalPreprocessor.letterboxToNchw(pcRgbImage)
            val exactInputComparison = compare(goldenInput, exactTensor.values, 0.0, 0.0)
            val exactOutput = infer(session, exactTensor.values)
            val exactOutputComparison = compare(goldenOutput, exactOutput, 1e-3, 1e-3)
            report.put("exact_from_shared_rgb8", JSONObject().apply {
                put("source_rgb_sha256_android", sha256(pcSourceRgb))
                put("geometry", geometryJson(exactTensor.geometry))
                put("tensor_vs_pc", exactInputComparison)
                put("output_vs_pc", exactOutputComparison)
            })

            // Functional test: each platform decodes the same JPEG with its native decoder,
            // then both use the canonical deterministic letterbox implementation.
            val decodeStart = SystemClock.elapsedRealtimeNanos()
            val androidRgb = CanonicalPreprocessor.decodeJpegSrgb(imageBytes)
            val decodeMs = elapsedMs(decodeStart)
            check(androidRgb.width == sourceWidth && androidRgb.height == sourceHeight) {
                "Dimensiones JPEG distintas: Android=${androidRgb.width}x${androidRgb.height}, PC=${sourceWidth}x${sourceHeight}"
            }
            val tensorStart = SystemClock.elapsedRealtimeNanos()
            val jpegTensor = CanonicalPreprocessor.letterboxToNchw(androidRgb)
            val preprocessingMs = elapsedMs(tensorStart)
            val inferenceStart = SystemClock.elapsedRealtimeNanos()
            val jpegOutput = infer(session, jpegTensor.values)
            val inferenceMs = elapsedMs(inferenceStart)
            val candidates = decodeCandidates(jpegOutput)
            val detections = classAwareNms(candidates, IOU)
            report.put("jpeg_native_decode_functional", JSONObject().apply {
                put("android_decoder", "BitmapFactory ARGB_8888, explicit sRGB, inScaled=false")
                put("decoded_rgb_vs_pc", compareRgb8(pcSourceRgb, androidRgb.rgb))
                put("geometry", geometryJson(jpegTensor.geometry))
                put("tensor_exact_vs_pc", compare(goldenInput, jpegTensor.values, 0.0, 0.0))
                put("tensor_within_1_u8_vs_pc", compare(goldenInput, jpegTensor.values, 0.0, 1.0 / 255.0 + 1e-8))
                put("tensor_within_2_u8_vs_pc", compare(goldenInput, jpegTensor.values, 0.0, 2.0 / 255.0 + 1e-8))
                put("tensor_within_3_u8_vs_pc", compare(goldenInput, jpegTensor.values, 0.0, 3.0 / 255.0 + 1e-8))
                put("output_vs_pc", compare(goldenOutput, jpegOutput, 1e-3, 1e-3))
                put("candidate_count", candidates.size)
                put("detections", detectionsJson(detections))
                put("timing_ms", JSONObject().apply {
                    put("jpeg_decode", decodeMs)
                    put("canonical_letterbox_and_tensor", preprocessingMs)
                    put("onnx_inference", inferenceMs)
                })
            })

            val exactPass = exactInputComparison.getBoolean("passed") && exactOutputComparison.getBoolean("passed")
            report.put("status", if (exactPass) "PASS_CANONICAL_EXACT" else "FAIL_CANONICAL_EXACT")
            report.put("equivalence_definition", JSONObject().apply {
                put("exact", "Identical RGB8 source -> bit-identical tensor; ONNX output within fixed FP32 tolerance")
                put("functional", "Same JPEG -> native decoders may differ; geometry/channel/normalization remain identical and differences are reported, not hidden")
            })
            val saved = saveNamedReport("preprocessing_validation", report)
            reportLine("${report.getString("status")}: exact tensor failed=${exactInputComparison.getInt("failed_elements")}")
            reportLine("Resultado: ${saved.absolutePath}")
            return report
        } catch (error: Throwable) {
            report.put("status", "ABORTED")
            report.put("error", error.stackTraceToString())
            saveNamedReport("preprocessing_validation", report)
            throw error
        } finally {
            session.close()
            options.close()
        }
    }

    /** One NNAPI inference with the locked canonical JPEG pipeline; no warmups or benchmark. */
    fun runNnapiValidation(): JSONObject {
        check(cpuGateFile(context).isFile) {
            "NNAPI bloqueado: falta el PASS CPU/XNNPACK"
        }
        val report = baseReport(Backend.NNAPI).apply {
            put("phase", "nnapi_short_validation")
            put("warmups_requested", 0)
            put("measured_runs_requested", 1)
            put("canonical_pipeline_locked", true)
        }
        val modelFile = File(context.noBackupFilesDir, MODEL_ASSET)
        context.assets.open(MODEL_ASSET).use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }
        val deviceHash = sha256(modelFile)
        report.put("model_file_bytes", modelFile.length())
        report.put("model_sha256_device", deviceHash)
        check(deviceHash.equals(MODEL_SHA256, ignoreCase = true)) {
            "SHA-256 del ONNX no coincide: $deviceHash"
        }

        val canonicalMetadata = JSONObject(
            context.assets.open(CANONICAL_META_ASSET).bufferedReader().use { it.readText() }
        )
        val goldenInput = readFloatAsset(CANONICAL_INPUT_ASSET, CHANNELS * SIZE * SIZE)
        val goldenOutput = readFloatAsset(CANONICAL_OUTPUT_ASSET, OUTPUT_CHANNELS * ANCHORS)
        val imageBytes = context.assets.open(IMAGE_ASSET).use { it.readBytes() }
        report.put("canonical_metadata", canonicalMetadata)

        val decodeStart = SystemClock.elapsedRealtimeNanos()
        val decoded = CanonicalPreprocessor.decodeJpegSrgb(imageBytes)
        val decodeMs = elapsedMs(decodeStart)
        val letterboxStart = SystemClock.elapsedRealtimeNanos()
        val letterboxed = CanonicalPreprocessor.letterboxRgb8(decoded)
        val letterboxMs = elapsedMs(letterboxStart)
        val tensorStart = SystemClock.elapsedRealtimeNanos()
        val input = CanonicalPreprocessor.rgbToNchw(letterboxed.image)
        val tensorMs = elapsedMs(tensorStart)
        val inputComparison = compare(goldenInput, input, 0.0, 0.0)
        report.put("canonical_preprocessing", JSONObject().apply {
            put("geometry", geometryJson(letterboxed.geometry))
            put("tensor_vs_pc", inputComparison)
            put("decode_ms", decodeMs)
            put("letterbox_ms", letterboxMs)
            put("tensor_ms", tensorMs)
            put("total_ms", decodeMs + letterboxMs + tensorMs)
        })
        check(inputComparison.getBoolean("passed")) {
            "El pipeline canónico dejó de ser exacto; NNAPI no se ejecuta"
        }

        val profilePrefix = File(context.filesDir, "ort_profile_nnapi_validation").absolutePath
        val options = sessionOptions(Backend.NNAPI, profiling = true, profilePrefix = profilePrefix)
        var session: OrtSession? = null
        try {
            val loadStart = SystemClock.elapsedRealtimeNanos()
            session = environment.createSession(modelFile.absolutePath, options)
            report.put("session_load_ms", elapsedMs(loadStart))
            report.put("model_io", validateModelIo(session))
            val inferenceStart = SystemClock.elapsedRealtimeNanos()
            val output = infer(session, input)
            report.put("inference_ms", elapsedMs(inferenceStart))
            val comparison = compare(goldenOutput, output, 1e-3, 1e-3)
            report.put("output_vs_pc", comparison)
            val candidates = decodeCandidates(output)
            report.put("candidate_count", candidates.size)
            report.put("detections", detectionsJson(classAwareNms(candidates, IOU)))
            val profilePath = session.endProfiling()
            report.put("provider_profile", parseProviderProfile(File(profilePath), profilePath))
            if (!comparison.getBoolean("passed")) {
                report.put("status", "FAIL_GOLDEN_NNAPI")
                saveNamedReport("nnapi_validation", report)
                error("NNAPI no cumple rtol=1e-3, atol=1e-3; no se habilita el benchmark")
            }
            report.put("status", "PASS_NNAPI_VALIDATION")
            val saved = saveNamedReport("nnapi_validation", report)
            nnapiValidationGateFile(context).writeText(JSONObject().apply {
                put("status", "PASS_NNAPI_VALIDATION")
                put("model_sha256", MODEL_SHA256)
                put("report", saved.name)
            }.toString(2))
            reportLine("PASS NNAPI validación corta: max_abs=${comparison.getDouble("max_abs")}, failed=0")
            reportLine("Resultado: ${saved.absolutePath}")
            return report
        } catch (error: Throwable) {
            if (!report.has("status")) report.put("status", "ABORTED")
            report.put("error", error.stackTraceToString())
            saveNamedReport("nnapi_validation", report)
            throw error
        } finally {
            session?.close()
            options.close()
        }
    }

    fun run(backend: Backend): JSONObject {
        if (backend == Backend.NNAPI && !cpuGateFile(context).isFile) {
            error("NNAPI bloqueado: primero debe existir un resultado PASS de CPU/XNNPACK")
        }
        if (backend == Backend.NNAPI && !nnapiValidationGateFile(context).isFile) {
            error("Benchmark NNAPI bloqueado: primero debe pasar la validación corta NNAPI")
        }
        if (isCharging()) {
            val blocked = baseReport(backend).apply {
                put("status", "BLOCKED_CHARGING")
                put("message", "Desconecte el cargador antes de una prueba sostenida")
                put("battery", batterySnapshot())
            }
            saveReport(backend, blocked)
            error("El teléfono está cargando. Benchmark sostenido cancelado.")
        }

        val report = baseReport(backend)
        val modelFile = File(context.noBackupFilesDir, MODEL_ASSET)
        val copyStart = SystemClock.elapsedRealtimeNanos()
        context.assets.open(MODEL_ASSET).use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }
        val modelCopyMs = elapsedMs(copyStart)
        val deviceHash = sha256(modelFile)
        report.put("model_asset_copy_ms", modelCopyMs)
        report.put("model_file_bytes", modelFile.length())
        report.put("model_sha256_device", deviceHash)
        check(deviceHash.equals(MODEL_SHA256, ignoreCase = true)) {
            "SHA-256 del ONNX no coincide: $deviceHash"
        }

        val imageBytes = context.assets.open(IMAGE_ASSET).use { it.readBytes() }
        val goldenInput = readFloatAsset(CANONICAL_INPUT_ASSET, CHANNELS * SIZE * SIZE)
        val goldenOutput = readFloatAsset(CANONICAL_OUTPUT_ASSET, OUTPUT_CHANNELS * ANCHORS)
        val goldenMetadata = JSONObject(context.assets.open(CANONICAL_META_ASSET).bufferedReader().use { it.readText() })
        report.put("golden_metadata", goldenMetadata)
        report.put("preprocessing_pipeline", "canonical_v1_locked")
        report.put("memory_before_session", memorySnapshot())
        report.put("thermal_before_session", thermalSnapshot())

        val sessionOptions = sessionOptions(backend, profiling = false)
        val loadStart = SystemClock.elapsedRealtimeNanos()
        val session = environment.createSession(modelFile.absolutePath, sessionOptions)
        val sessionLoadMs = elapsedMs(loadStart)
        report.put("session_load_ms", sessionLoadMs)
        report.put("memory_after_session", memorySnapshot())
        report.put("thermal_after_session", thermalSnapshot())

        try {
            report.put("model_io", validateModelIo(session))

            val isolatedStart = SystemClock.elapsedRealtimeNanos()
            val isolatedOutput = infer(session, goldenInput)
            val isolatedMs = elapsedMs(isolatedStart)
            val isolatedComparison = compare(goldenOutput, isolatedOutput, 1e-3, 1e-3)
            report.put("golden_tensor_runtime_inference_ms", isolatedMs)
            report.put("golden_tensor_runtime_comparison", isolatedComparison)
            reportLine("Golden tensor -> ORT: max_abs=${isolatedComparison.getDouble("max_abs")}, " +
                "failed=${isolatedComparison.getInt("failed_elements")}")
            if (!isolatedComparison.getBoolean("passed")) {
                report.put("status", "FAIL_GOLDEN_RUNTIME")
                report.put("memory_after", memorySnapshot())
                report.put("thermal_after", thermalSnapshot())
                error("Salida ORT relevante respecto al golden; no se ejecutan 100 iteraciones")
            }

            val first = runPipeline(session, imageBytes, captureInput = true)
            val inputComparison = compare(goldenInput, first.input!!, 0.0, 0.0)
            val imageOutputComparison = compare(goldenOutput, first.output, 1e-3, 1e-3)
            val goldenDetections = goldenMetadata.getJSONArray("detections")
            val detectionComparison = compareDetections(goldenDetections, first.detections)
            report.put("first_image_pipeline", JSONObject().apply {
                put("times_ms", stageTimesJson(first.times))
                put("input_vs_pc", inputComparison)
                put("output_vs_pc", imageOutputComparison)
                put("candidate_count", first.candidates)
                put("detections", detectionsJson(first.detections))
                put("detections_vs_pc", detectionComparison)
                put("representative_output_values", representativeValues(first.output))
            })
            reportLine("Imagen -> tensor: max_abs=${inputComparison.getDouble("max_abs")}")
            reportLine("Imagen -> ORT: max_abs=${imageOutputComparison.getDouble("max_abs")}")
            reportLine("Detecciones: ${first.detections.size}")
            if (!inputComparison.getBoolean("passed") || !imageOutputComparison.getBoolean("passed")) {
                report.put("status", "FAIL_CANONICAL_PRE_BENCHMARK")
                error("Validación canónica previa falló; no se ejecutan warmups ni 100 iteraciones")
            }

            val warmupTimes = JSONArray()
            repeat(WARMUPS) { index ->
                ensureSafeToContinue("warmup_${index + 1}")
                val value = runPipeline(session, imageBytes)
                warmupTimes.put(value.times.inferenceMs)
                reportLine("warmup ${index + 1}/$WARMUPS inference=${fmt(value.times.inferenceMs)} ms")
            }
            report.put("warmup_inference_ms", warmupTimes)
            report.put("thermal_after_warmup", thermalSnapshot())

            val measured = mutableListOf<StageTimes>()
            val thermalEvolution = JSONArray()
            val memoryEvolution = JSONArray()
            repeat(MEASURED_RUNS) { index ->
                ensureSafeToContinue("measured_${index + 1}")
                val value = runPipeline(session, imageBytes)
                measured += value.times
                reportLine(
                    "run ${index + 1}/$MEASURED_RUNS inference=${fmt(value.times.inferenceMs)} " +
                        "e2e=${fmt(value.times.totalMs)} ms"
                )
                if ((index + 1) % 10 == 0) {
                    thermalEvolution.put(JSONObject().apply {
                        put("run", index + 1)
                        put("snapshot", thermalSnapshot())
                    })
                    memoryEvolution.put(JSONObject().apply {
                        put("run", index + 1)
                        put("snapshot", memorySnapshot())
                    })
                }
            }

            report.put("measured_runs_completed", measured.size)
            report.put("individual_times_ms", individualTimesJson(measured))
            report.put("summary", summaryJson(measured, sessionLoadMs))
            report.put("thermal_evolution", thermalEvolution)
            report.put("memory_evolution", memoryEvolution)
            report.put("memory_after_benchmark_before_profile", memorySnapshot())
            report.put("thermal_after_benchmark_before_profile", thermalSnapshot())

            // Run provider profiling only after all latency measurements, in a
            // separate session, so profiler overhead cannot affect the benchmark.
            report.put("provider_profile", runProviderDiagnostic(backend, modelFile, goldenInput))
            System.gc()
            report.put("memory_after", memorySnapshot())
            report.put("thermal_after", thermalSnapshot())
            report.put("battery_after", batterySnapshot())
            report.put("status", "PASS")

            val saved = saveReport(backend, report)
            if (backend == Backend.XNNPACK) {
                cpuGateFile(context).writeText(JSONObject().apply {
                    put("status", "PASS")
                    put("model_sha256", MODEL_SHA256)
                    put("report", saved.name)
                }.toString(2))
            }
            val inferenceMean = report.getJSONObject("summary").getJSONObject("inference_ms").getDouble("mean")
            reportLine("PASS ${backend.label}: mean=${fmt(inferenceMean)} ms, FPS=${fmt(1000.0 / inferenceMean)}")
            reportLine("Resultado: ${saved.absolutePath}")
            return report
        } catch (error: Throwable) {
            if (!report.has("status")) report.put("status", "ABORTED")
            report.put("error", error.stackTraceToString())
            report.put("memory_after", memorySnapshot())
            report.put("thermal_after", thermalSnapshot())
            saveReport(backend, report)
            throw error
        } finally {
            session.close()
            sessionOptions.close()
        }
    }

    private fun runProviderDiagnostic(backend: Backend, modelFile: File, goldenInput: FloatArray): JSONObject {
        val prefix = File(context.filesDir, "ort_profile_${backend.wireName}").absolutePath
        val options = sessionOptions(backend, profiling = true, profilePrefix = prefix)
        val session = environment.createSession(modelFile.absolutePath, options)
        return try {
            infer(session, goldenInput)
            val profilePath = session.endProfiling()
            val file = File(profilePath)
            parseProviderProfile(file, profilePath)
        } finally {
            session.close()
            options.close()
        }
    }

    private fun parseProviderProfile(file: File, profilePath: String): JSONObject {
        val eventCounts = sortedMapOf<String, Int>()
        val durationUs = sortedMapOf<String, Long>()
        val nodeIndices = sortedMapOf<String, MutableSet<String>>()
        val nodeNames = sortedMapOf<String, MutableSet<String>>()
        val opTypes = sortedMapOf<String, MutableSet<String>>()
        val providerSequence = mutableListOf<String>()
        if (file.isFile) {
            val events = JSONArray(file.readText())
            for (index in 0 until events.length()) {
                val event = events.optJSONObject(index) ?: continue
                if (event.optString("cat") != "Node") continue
                val args = event.optJSONObject("args") ?: continue
                val provider = args.optString("provider")
                if (provider.isBlank()) continue
                eventCounts[provider] = (eventCounts[provider] ?: 0) + 1
                durationUs[provider] = (durationUs[provider] ?: 0L) + event.optLong("dur", 0L)
                args.optString("node_index").takeIf { it.isNotBlank() }
                    ?.let { nodeIndices.getOrPut(provider) { sortedSetOf() }.add(it) }
                event.optString("name").removeSuffix("_kernel_time").takeIf { it.isNotBlank() }
                    ?.let { nodeNames.getOrPut(provider) { sortedSetOf() }.add(it) }
                args.optString("op_name").takeIf { it.isNotBlank() }
                    ?.let { opTypes.getOrPut(provider) { sortedSetOf() }.add(it) }
                if (providerSequence.lastOrNull() != provider) providerSequence += provider
            }
        }
        fun setMapJson(values: Map<String, Set<String>>) = JSONObject().apply {
            values.forEach { (provider, items) -> put(provider, JSONArray(items.toList())) }
        }
        return JSONObject().apply {
            put("profile_file", profilePath)
            put("profile_file_exists", file.isFile)
            put("provider_event_counts", JSONObject(eventCounts as Map<*, *>))
            put("kernel_duration_us_by_provider", JSONObject(durationUs as Map<*, *>))
            put("unique_profiled_node_count_by_provider", JSONObject(
                nodeIndices.mapValues { it.value.size } as Map<*, *>
            ))
            put("unique_profiled_node_names_by_provider", setMapJson(nodeNames))
            put("op_types_by_provider", setMapJson(opTypes))
            put("provider_sequence_collapsed", JSONArray(providerSequence))
            put("cpu_fallback_evidence", JSONObject().apply {
                val cpu = "CPUExecutionProvider"
                put("profiled_node_count", nodeIndices[cpu]?.size ?: 0)
                put("op_types", JSONArray(opTypes[cpu]?.toList() ?: emptyList<String>()))
                put("node_names", JSONArray(nodeNames[cpu]?.toList() ?: emptyList<String>()))
            })
            put("note", "Event counts and durations are profiler evidence only; they are not percentages of graph nodes or workload")
        }
    }

    private fun sessionOptions(
        backend: Backend,
        profiling: Boolean,
        profilePrefix: String = ""
    ): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
        setMemoryPatternOptimization(true)
        if (profiling) enableProfiling(profilePrefix)
        when (backend) {
            Backend.XNNPACK -> {
                addConfigEntry("session.intra_op.allow_spinning", "0")
                setIntraOpNumThreads(1)
                addXnnpack(mapOf("intra_op_num_threads" to XNNPACK_THREADS.toString()))
            }
            Backend.NNAPI -> {
                // No USE_FP16: this baseline remains FP32. CPU_DISABLED prevents
                // NNAPI's own CPU implementation; ORT CPU remains the fallback EP.
                addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
            }
        }
    }

    private fun validateModelIo(session: OrtSession): JSONObject {
        val inputInfo = session.inputInfo["images"]?.info as? TensorInfo
            ?: error("No existe la entrada 'images'")
        val outputName = session.outputNames.singleOrNull() ?: error("Se esperaba una sola salida")
        val outputInfo = session.outputInfo[outputName]?.info as? TensorInfo
            ?: error("Salida sin TensorInfo")
        check(inputInfo.shape.contentEquals(longArrayOf(1, 3, 640, 640))) { "Input inesperado: ${inputInfo.shape.contentToString()}" }
        check(outputInfo.shape.contentEquals(longArrayOf(1, 8, 8400))) { "Output inesperado: ${outputInfo.shape.contentToString()}" }
        return JSONObject().apply {
            put("input_name", "images")
            put("input_shape", JSONArray(inputInfo.shape.toList()))
            put("output_name", outputName)
            put("output_shape", JSONArray(outputInfo.shape.toList()))
        }
    }

    private fun runPipeline(session: OrtSession, imageBytes: ByteArray, captureInput: Boolean = false): PipelineResult {
        val totalStart = SystemClock.elapsedRealtimeNanos()

        val decodeStart = SystemClock.elapsedRealtimeNanos()
        val decoded = CanonicalPreprocessor.decodeJpegSrgb(imageBytes)
        val decodeMs = elapsedMs(decodeStart)

        val resizeStart = SystemClock.elapsedRealtimeNanos()
        val resized = CanonicalPreprocessor.letterboxRgb8(decoded)
        val resizeMs = elapsedMs(resizeStart)

        val tensorStart = SystemClock.elapsedRealtimeNanos()
        val input = CanonicalPreprocessor.rgbToNchw(resized.image)
        val tensorMs = elapsedMs(tensorStart)

        val inferenceStart = SystemClock.elapsedRealtimeNanos()
        val output = infer(session, input)
        val inferenceMs = elapsedMs(inferenceStart)

        val postStart = SystemClock.elapsedRealtimeNanos()
        val candidates = decodeCandidates(output)
        val postMs = elapsedMs(postStart)

        val nmsStart = SystemClock.elapsedRealtimeNanos()
        val detections = classAwareNms(candidates, IOU)
        val nmsMs = elapsedMs(nmsStart)

        return PipelineResult(
            StageTimes(decodeMs, resizeMs, tensorMs, inferenceMs, postMs, nmsMs, elapsedMs(totalStart)),
            output,
            candidates.size,
            detections,
            if (captureInput) input else null
        )
    }

    private fun infer(session: OrtSession, input: FloatArray): FloatArray {
        val byteBuffer = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()
        floatBuffer.put(input).rewind()
        OnnxTensor.createTensor(environment, floatBuffer, longArrayOf(1, 3, 640, 640)).use { tensor ->
            session.run(mapOf("images" to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val value = result[0].value as Array<Array<FloatArray>>
                check(value.size == 1 && value[0].size == OUTPUT_CHANNELS && value[0][0].size == ANCHORS) {
                    "Forma de salida inesperada"
                }
                val flat = FloatArray(OUTPUT_CHANNELS * ANCHORS)
                for (channel in 0 until OUTPUT_CHANNELS) {
                    System.arraycopy(value[0][channel], 0, flat, channel * ANCHORS, ANCHORS)
                }
                return flat
            }
        }
    }

    private fun decodeCandidates(output: FloatArray): List<Detection> {
        val detections = ArrayList<Detection>()
        for (anchor in 0 until ANCHORS) {
            var bestClass = 0
            var bestScore = output[4 * ANCHORS + anchor]
            for (classId in 1 until CLASS_NAMES.size) {
                val score = output[(4 + classId) * ANCHORS + anchor]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classId
                }
            }
            if (bestScore < CONFIDENCE) continue
            val cx = output[anchor]
            val cy = output[ANCHORS + anchor]
            val width = output[2 * ANCHORS + anchor]
            val height = output[3 * ANCHORS + anchor]
            detections += Detection(
                bestClass,
                bestScore,
                cx - width / 2f,
                cy - height / 2f,
                cx + width / 2f,
                cy + height / 2f
            )
        }
        return detections
    }

    private fun classAwareNms(candidates: List<Detection>, threshold: Float): List<Detection> {
        val kept = ArrayList<Detection>()
        for (classId in CLASS_NAMES.indices) {
            val pending = candidates.filter { it.classId == classId }
                .sortedByDescending { it.score }.toMutableList()
            while (pending.isNotEmpty()) {
                val best = pending.removeAt(0)
                kept += best
                pending.removeAll { iou(best, it) > threshold }
            }
        }
        return kept.sortedByDescending { it.score }
    }

    private fun iou(a: Detection, b: Detection): Float {
        val intersectionWidth = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val intersectionHeight = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val intersection = intersectionWidth * intersectionHeight
        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        return intersection / max(areaA + areaB - intersection, 1e-9f)
    }

    private fun compare(reference: FloatArray, candidate: FloatArray, rtol: Double, atol: Double): JSONObject {
        check(reference.size == candidate.size)
        var maxAbs = 0.0
        var sumAbs = 0.0
        var maxRel = 0.0
        var sumRel = 0.0
        var failed = 0
        for (index in reference.indices) {
            val ref = reference[index].toDouble()
            val diff = kotlin.math.abs(ref - candidate[index].toDouble())
            val rel = diff / max(kotlin.math.abs(ref), 1e-8)
            maxAbs = max(maxAbs, diff)
            maxRel = max(maxRel, rel)
            sumAbs += diff
            sumRel += rel
            if (diff > atol + rtol * kotlin.math.abs(ref)) failed++
        }
        return JSONObject().apply {
            put("elements", reference.size)
            put("rtol", rtol)
            put("atol", atol)
            put("max_abs", maxAbs)
            put("mean_abs", sumAbs / reference.size)
            put("max_rel_eps_1e-8", maxRel)
            put("mean_rel_eps_1e-8", sumRel / reference.size)
            put("failed_elements", failed)
            put("passed", failed == 0)
        }
    }

    private fun compareRgb8(reference: ByteArray, candidate: ByteArray): JSONObject {
        check(reference.size == candidate.size)
        var different = 0
        var withinOne = 0
        var withinTwo = 0
        var withinThree = 0
        var maxAbs = 0
        var sumAbs = 0L
        for (index in reference.indices) {
            val delta = kotlin.math.abs((reference[index].toInt() and 0xff) - (candidate[index].toInt() and 0xff))
            if (delta != 0) different++
            if (delta <= 1) withinOne++
            if (delta <= 2) withinTwo++
            if (delta <= 3) withinThree++
            maxAbs = max(maxAbs, delta)
            sumAbs += delta
        }
        return JSONObject().apply {
            put("elements", reference.size)
            put("different_elements", different)
            put("max_abs_u8", maxAbs)
            put("mean_abs_u8", sumAbs.toDouble() / reference.size)
            put("within_1_u8_elements", withinOne)
            put("within_2_u8_elements", withinTwo)
            put("within_3_u8_elements", withinThree)
        }
    }

    private fun geometryJson(value: LetterboxGeometry) = JSONObject().apply {
        put("source_width", value.sourceWidth)
        put("source_height", value.sourceHeight)
        put("target_width", value.targetWidth)
        put("target_height", value.targetHeight)
        put("resized_width", value.resizedWidth)
        put("resized_height", value.resizedHeight)
        put("pad_left", value.padLeft)
        put("pad_right", value.padRight)
        put("pad_top", value.padTop)
        put("pad_bottom", value.padBottom)
        put("padding_rgb", JSONArray(listOf(114, 114, 114)))
    }

    private fun compareDetections(golden: JSONArray, actual: List<Detection>): JSONObject {
        var maxScoreAbs = 0.0
        var maxBoxAbs = 0.0
        val matched = min(golden.length(), actual.size)
        var classMismatches = 0
        for (index in 0 until matched) {
            val expected = golden.getJSONObject(index)
            val observed = actual[index]
            if (expected.getInt("class_id") != observed.classId) classMismatches++
            maxScoreAbs = max(maxScoreAbs, kotlin.math.abs(expected.getDouble("score") - observed.score))
            val box = expected.getJSONArray("xyxy")
            val values = doubleArrayOf(observed.x1.toDouble(), observed.y1.toDouble(), observed.x2.toDouble(), observed.y2.toDouble())
            for (coordinate in 0..3) maxBoxAbs = max(maxBoxAbs, kotlin.math.abs(box.getDouble(coordinate) - values[coordinate]))
        }
        return JSONObject().apply {
            put("golden_count", golden.length())
            put("actual_count", actual.size)
            put("matched_by_rank", matched)
            put("class_mismatches", classMismatches)
            put("max_score_abs", maxScoreAbs)
            put("max_box_abs_pixels", maxBoxAbs)
        }
    }

    private fun summaryJson(values: List<StageTimes>, sessionLoadMs: Double): JSONObject {
        fun field(selector: (StageTimes) -> Double) = stats(values.map(selector))
        val inference = values.map { it.inferenceMs }
        val window = min(20, max(1, values.size / 2))
        val firstWindow = inference.take(window).average()
        val lastWindow = inference.takeLast(window).average()
        return JSONObject().apply {
            put("session_load_ms", sessionLoadMs)
            put("decode_ms", field { it.decodeMs })
            put("resize_ms", field { it.resizeMs })
            put("tensor_ms", field { it.tensorMs })
            put("preprocessing_ms", field { it.preprocessingMs })
            put("inference_ms", stats(inference))
            put("fps_equivalent_from_inference_mean", 1000.0 / inference.average())
            put("inference_stability", JSONObject().apply {
                put("window_runs", window)
                put("first_window_mean_ms", firstWindow)
                put("last_window_mean_ms", lastWindow)
                put("last_over_first_ratio", lastWindow / firstWindow)
                put("possible_sustained_slowdown_over_15_percent", lastWindow > firstWindow * 1.15)
            })
            put("postprocess_ms", field { it.postprocessMs })
            put("nms_ms", field { it.nmsMs })
            put("end_to_end_ms", field { it.totalMs })
        }
    }

    private fun stats(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        fun percentile(p: Double): Double = sorted[(ceil(p * sorted.size).toInt() - 1).coerceIn(0, sorted.lastIndex)]
        val median = if (sorted.size % 2 == 0) {
            (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
        } else sorted[sorted.size / 2]
        return JSONObject().apply {
            put("count", values.size)
            put("mean", values.average())
            put("median", median)
            put("p50", percentile(0.50))
            put("p90", percentile(0.90))
            put("p95", percentile(0.95))
            put("min", sorted.first())
            put("max", sorted.last())
        }
    }

    private fun individualTimesJson(values: List<StageTimes>) = JSONObject().apply {
        put("decode", JSONArray(values.map { it.decodeMs }))
        put("resize", JSONArray(values.map { it.resizeMs }))
        put("tensor", JSONArray(values.map { it.tensorMs }))
        put("preprocessing", JSONArray(values.map { it.preprocessingMs }))
        put("inference", JSONArray(values.map { it.inferenceMs }))
        put("postprocess", JSONArray(values.map { it.postprocessMs }))
        put("nms", JSONArray(values.map { it.nmsMs }))
        put("end_to_end", JSONArray(values.map { it.totalMs }))
    }

    private fun stageTimesJson(value: StageTimes) = JSONObject().apply {
        put("decode", value.decodeMs)
        put("resize", value.resizeMs)
        put("tensor", value.tensorMs)
        put("preprocessing", value.preprocessingMs)
        put("inference", value.inferenceMs)
        put("postprocess", value.postprocessMs)
        put("nms", value.nmsMs)
        put("end_to_end", value.totalMs)
    }

    private fun detectionsJson(values: List<Detection>) = JSONArray().apply {
        values.forEach { detection ->
            put(JSONObject().apply {
                put("class_id", detection.classId)
                put("class_name", CLASS_NAMES[detection.classId])
                put("score", detection.score.toDouble())
                put("xyxy", JSONArray(listOf(detection.x1, detection.y1, detection.x2, detection.y2)))
            })
        }
    }

    private fun representativeValues(output: FloatArray) = JSONArray().apply {
        val indices = intArrayOf(0, 1, 100, 8399, 8400, 16800, 25200, 33600, 42000, 50400, 58800, 67199)
        indices.forEach { index -> put(JSONObject().apply { put("index", index); put("value", output[index].toDouble()) }) }
    }

    private fun readFloatAsset(name: String, expectedFloats: Int): FloatArray {
        val bytes = context.assets.open(name).use { it.readBytes() }
        check(bytes.size == expectedFloats * 4) { "$name tiene ${bytes.size} bytes; se esperaban ${expectedFloats * 4}" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(expectedFloats).also { buffer.get(it) }
    }

    private fun baseReport(backend: Backend) = JSONObject().apply {
        put("schema_version", 1)
        put("started_utc", Instant.now().toString())
        put("backend", backend.wireName)
        put("backend_label", backend.label)
        put("phase", if (backend == Backend.XNNPACK) 1 else 2)
        put("model_asset", MODEL_ASSET)
        put("model_sha256_expected", MODEL_SHA256)
        put("precision", "FP32")
        put("input_shape", JSONArray(listOf(1, 3, 640, 640)))
        put("output_shape", JSONArray(listOf(1, 8, 8400)))
        put("warmups_requested", WARMUPS)
        put("measured_runs_requested", MEASURED_RUNS)
        put("confidence_threshold", CONFIDENCE.toDouble())
        put("iou_threshold", IOU.toDouble())
        put("onnxruntime_android_version", BuildConfig.ORT_VERSION)
        put("xnnpack_threads", if (backend == Backend.XNNPACK) XNNPACK_THREADS else JSONObject.NULL)
        put("device", JSONObject().apply {
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("device", Build.DEVICE)
            put("hardware", Build.HARDWARE)
            put("soc_manufacturer", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else "unavailable")
            put("soc_model", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unavailable")
            put("android_release", Build.VERSION.RELEASE)
            put("sdk_int", Build.VERSION.SDK_INT)
            put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            put("available_processors", Runtime.getRuntime().availableProcessors())
        })
        put("available_ort_providers", runCatching { JSONArray(OrtEnvironment.getAvailableProviders().toList()) }.getOrElse { JSONArray().put("unavailable: ${it.message}") })
        put("battery_before", batterySnapshot())
    }

    private fun memorySnapshot(): JSONObject {
        val debug = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
        val runtime = Runtime.getRuntime()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return JSONObject().apply {
            put("total_pss_kb", debug.totalPss)
            put("total_private_dirty_kb", debug.totalPrivateDirty)
            put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
            put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
            put("java_heap_max_bytes", runtime.maxMemory())
            put("memory_class_mb", activityManager.memoryClass)
        }
    }

    private fun thermalSnapshot(): JSONObject {
        val manager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val status = if (Build.VERSION.SDK_INT >= 29) manager.currentThermalStatus else -1
        return JSONObject().apply {
            put("status_code", status)
            put("status_name", thermalName(status))
            put("source", "Android PowerManager thermal status; not an SoC temperature")
            put("elapsed_realtime_ms", SystemClock.elapsedRealtime())
        }
    }

    private fun batterySnapshot(): JSONObject {
        val state = context.registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val plugged = state?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val rawTemperature = state?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        return JSONObject().apply {
            put("charging_or_plugged", plugged != 0)
            put("plugged_code", plugged)
            put("level_percent", state?.let {
                val level = it.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = it.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (level >= 0 && scale > 0) 100.0 * level / scale else JSONObject.NULL
            } ?: JSONObject.NULL)
            put("battery_temperature_c", if (rawTemperature != Int.MIN_VALUE) rawTemperature / 10.0 else JSONObject.NULL)
            put("temperature_warning", "Battery sensor only; do not interpret as SoC temperature")
        }
    }

    private fun isCharging(): Boolean = batterySnapshot().getBoolean("charging_or_plugged")

    private fun ensureSafeToContinue(stage: String) {
        check(!Thread.currentThread().isInterrupted) { "Interrumpido durante $stage" }
        check(!isCharging()) { "El teléfono comenzó a cargar durante $stage" }
        val thermal = thermalSnapshot().getInt("status_code")
        check(thermal < PowerManager.THERMAL_STATUS_SEVERE || thermal < 0) {
            "Estado térmico ${thermalName(thermal)} durante $stage; prueba abortada"
        }
    }

    private fun thermalName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "NONE"
        PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
        PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
        PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
        PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
        else -> "UNAVAILABLE"
    }

    private fun saveReport(backend: Backend, report: JSONObject): File {
        return saveNamedReport(backend.wireName, report)
    }

    private fun saveNamedReport(prefix: String, report: JSONObject): File {
        val directory = File(context.filesDir, "results").apply { mkdirs() }
        val file = File(directory, "${prefix}_${System.currentTimeMillis()}.json")
        file.writeText(report.toString(2))
        reportLine("REPORT_FILE=${file.absolutePath}")
        return file
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private fun elapsedMs(startNs: Long): Double = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
    private fun fmt(value: Double): String = String.format(Locale.US, "%.3f", value)
}
