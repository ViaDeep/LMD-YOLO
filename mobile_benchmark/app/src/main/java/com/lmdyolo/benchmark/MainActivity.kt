package com.lmdyolo.benchmark

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var output: TextView
    private lateinit var cpuButton: Button
    private lateinit var nnapiButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "LMD YOLO Benchmark"

        output = TextView(this).apply {
            textSize = 12f
            setPadding(24, 24, 24, 24)
            setTextIsSelectable(true)
        }
        cpuButton = Button(this).apply {
            text = "Ejecutar CPU/XNNPACK"
            setOnClickListener { runBenchmark(Backend.XNNPACK) }
        }
        nnapiButton = Button(this).apply {
            text = "Ejecutar NNAPI (solo tras PASS CPU)"
            isEnabled = BenchmarkRunner.cpuGateFile(this@MainActivity).isFile &&
                BenchmarkRunner.nnapiValidationGateFile(this@MainActivity).isFile
            setOnClickListener { runBenchmark(Backend.NNAPI) }
        }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            addView(cpuButton)
            addView(nnapiButton)
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(buttons)
            addView(
                ScrollView(this@MainActivity).apply { addView(output) },
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f)
            )
        }
        setContentView(root)

        val requested = intent.getStringExtra("backend")?.lowercase()
        when (requested) {
            "nnapi" -> runBenchmark(Backend.NNAPI)
            "nnapi_validate" -> runNnapiValidation()
            "preprocess" -> runPreprocessingValidation()
            "none" -> append("Listo. Seleccione un backend.\n")
            else -> runBenchmark(Backend.XNNPACK)
        }
    }

    private fun runNnapiValidation() {
        cpuButton.isEnabled = false
        nnapiButton.isEnabled = false
        output.text = ""
        append("Validación corta NNAPI: una inferencia, golden y profiler; sin benchmark sostenido.\n")
        executor.execute {
            val runner = BenchmarkRunner(this) { append(it + "\n") }
            val result = runCatching { runner.runNnapiValidation() }
            result.onFailure {
                Log.e(BenchmarkRunner.TAG, "NNAPI short validation failed", it)
                append("ERROR: ${it.stackTraceToString()}\n")
            }
            runOnUiThread {
                cpuButton.isEnabled = true
                nnapiButton.isEnabled = BenchmarkRunner.cpuGateFile(this).isFile &&
                    BenchmarkRunner.nnapiValidationGateFile(this).isFile
            }
        }
    }

    private fun runPreprocessingValidation() {
        cpuButton.isEnabled = false
        nnapiButton.isEnabled = false
        output.text = ""
        append("Validando preprocesamiento canónico con CPU/XNNPACK; NNAPI no se ejecutará.\n")
        executor.execute {
            val runner = BenchmarkRunner(this) { append(it + "\n") }
            val result = runCatching { runner.runPreprocessingValidation() }
            result.onFailure {
                Log.e(BenchmarkRunner.TAG, "Preprocessing validation failed", it)
                append("ERROR: ${it.stackTraceToString()}\n")
            }
            runOnUiThread {
                cpuButton.isEnabled = true
                nnapiButton.isEnabled = BenchmarkRunner.cpuGateFile(this).isFile &&
                    BenchmarkRunner.nnapiValidationGateFile(this).isFile
            }
        }
    }

    private fun runBenchmark(backend: Backend) {
        cpuButton.isEnabled = false
        nnapiButton.isEnabled = false
        output.text = ""
        append("Iniciando ${backend.label}. No desconecte el dispositivo.\n")
        executor.execute {
            val runner = BenchmarkRunner(this) { append(it + "\n") }
            val result = runCatching { runner.run(backend) }
            result.onFailure {
                Log.e(BenchmarkRunner.TAG, "Benchmark failed", it)
                append("ERROR: ${it.stackTraceToString()}\n")
            }
            runOnUiThread {
                cpuButton.isEnabled = true
                nnapiButton.isEnabled = BenchmarkRunner.cpuGateFile(this).isFile &&
                    BenchmarkRunner.nnapiValidationGateFile(this).isFile
            }
        }
    }

    private fun append(message: String) {
        Log.i(BenchmarkRunner.TAG, message.trimEnd())
        runOnUiThread {
            output.append(message)
            (output.parent as? ScrollView)?.post { (output.parent as ScrollView).fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }
}
