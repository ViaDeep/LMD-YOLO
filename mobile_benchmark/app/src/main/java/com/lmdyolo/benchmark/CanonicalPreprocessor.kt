package com.lmdyolo.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace

data class RgbImage(val width: Int, val height: Int, val rgb: ByteArray)

data class LetterboxGeometry(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val targetWidth: Int,
    val targetHeight: Int,
    val resizedWidth: Int,
    val resizedHeight: Int,
    val padLeft: Int,
    val padRight: Int,
    val padTop: Int,
    val padBottom: Int
)

data class CanonicalTensor(val values: FloatArray, val geometry: LetterboxGeometry)

data class CanonicalLetterbox(val image: RgbImage, val geometry: LetterboxGeometry)

/**
 * Canonical production preprocessing shared equation-for-equation with
 * scripts/generate_canonical_mobile_golden.py.
 *
 * Contract after decoding:
 *   upright opaque sRGB RGB8 -> centered 640x640 letterbox -> RGB NCHW FP32/255.
 * The bilinear kernel uses half-pixel centers, exact rational integer weights,
 * and round-half-up. This avoids platform graphics-library resize differences.
 */
object CanonicalPreprocessor {
    const val TARGET_WIDTH = 640
    const val TARGET_HEIGHT = 640
    const val PADDING_VALUE = 114

    fun decodeJpegSrgb(jpeg: ByteArray): RgbImage {
        val options = BitmapFactory.Options().apply {
            inScaled = false
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
        }
        val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, options)
            ?: error("Android no pudo decodificar el JPEG")
        return try {
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            val rgb = ByteArray(pixels.size * 3)
            for (index in pixels.indices) {
                val color = pixels[index]
                val base = index * 3
                rgb[base] = ((color ushr 16) and 0xff).toByte()
                rgb[base + 1] = ((color ushr 8) and 0xff).toByte()
                rgb[base + 2] = (color and 0xff).toByte()
            }
            RgbImage(bitmap.width, bitmap.height, rgb)
        } finally {
            bitmap.recycle()
        }
    }

    fun letterboxToNchw(image: RgbImage): CanonicalTensor {
        val letterboxed = letterboxRgb8(image)
        return CanonicalTensor(rgbToNchw(letterboxed.image), letterboxed.geometry)
    }

    /** Same canonical letterbox, exposed separately only for timing. */
    fun letterboxRgb8(image: RgbImage): CanonicalLetterbox {
        require(image.rgb.size == image.width * image.height * 3)
        val geometry = geometry(image.width, image.height)
        val resized = resizeBilinear(image, geometry.resizedWidth, geometry.resizedHeight)
        val canvas = ByteArray(TARGET_WIDTH * TARGET_HEIGHT * 3) { PADDING_VALUE.toByte() }
        for (y in 0 until geometry.resizedHeight) {
            val sourceOffset = y * geometry.resizedWidth * 3
            val targetOffset = ((y + geometry.padTop) * TARGET_WIDTH + geometry.padLeft) * 3
            resized.copyInto(canvas, targetOffset, sourceOffset, sourceOffset + geometry.resizedWidth * 3)
        }
        return CanonicalLetterbox(RgbImage(TARGET_WIDTH, TARGET_HEIGHT, canvas), geometry)
    }

    /** RGB HWC uint8 -> RGB NCHW FP32 / 255, with no geometry change. */
    fun rgbToNchw(image: RgbImage): FloatArray {
        require(image.rgb.size == image.width * image.height * 3)
        val plane = image.width * image.height
        val tensor = FloatArray(3 * plane)
        for (index in 0 until plane) {
            val sourceIndex = index * 3
            tensor[index] = (image.rgb[sourceIndex].toInt() and 0xff) / 255.0f
            tensor[plane + index] = (image.rgb[sourceIndex + 1].toInt() and 0xff) / 255.0f
            tensor[2 * plane + index] = (image.rgb[sourceIndex + 2].toInt() and 0xff) / 255.0f
        }
        return tensor
    }

    private fun geometry(sourceWidth: Int, sourceHeight: Int): LetterboxGeometry {
        require(sourceWidth > 0 && sourceHeight > 0)
        val resizedWidth: Int
        val resizedHeight: Int
        if (TARGET_HEIGHT.toLong() * sourceWidth <= TARGET_WIDTH.toLong() * sourceHeight) {
            resizedHeight = TARGET_HEIGHT
            resizedWidth = roundDivideTiesToEven(sourceWidth.toLong() * TARGET_HEIGHT, sourceHeight.toLong())
        } else {
            resizedWidth = TARGET_WIDTH
            resizedHeight = roundDivideTiesToEven(sourceHeight.toLong() * TARGET_WIDTH, sourceWidth.toLong())
        }
        val padWidth = TARGET_WIDTH - resizedWidth
        val padHeight = TARGET_HEIGHT - resizedHeight
        val left = padWidth / 2
        val top = padHeight / 2
        return LetterboxGeometry(
            sourceWidth, sourceHeight, TARGET_WIDTH, TARGET_HEIGHT,
            resizedWidth, resizedHeight,
            left, padWidth - left, top, padHeight - top
        )
    }

    private fun roundDivideTiesToEven(numerator: Long, denominator: Long): Int {
        val quotient = numerator / denominator
        val remainder = numerator % denominator
        val rounded = when {
            remainder * 2 < denominator -> quotient
            remainder * 2 > denominator -> quotient + 1
            quotient % 2L == 0L -> quotient
            else -> quotient + 1
        }
        return rounded.toInt()
    }

    private data class AxisPoint(
        val low: Int,
        val high: Int,
        val weightLow: Long,
        val weightHigh: Long,
        val denominator: Long
    )

    private fun axisMap(sourceLength: Int, targetLength: Int): Array<AxisPoint> {
        val denominator = 2L * targetLength
        return Array(targetLength) { target ->
            val position = (2L * target + 1L) * sourceLength - targetLength
            when {
                position <= 0L -> AxisPoint(0, 0, denominator, 0, denominator)
                position >= (sourceLength - 1L) * denominator -> {
                    AxisPoint(sourceLength - 1, sourceLength - 1, denominator, 0, denominator)
                }
                else -> {
                    val low = Math.floorDiv(position, denominator).toInt()
                    val remainder = position - low.toLong() * denominator
                    AxisPoint(low, low + 1, denominator - remainder, remainder, denominator)
                }
            }
        }
    }

    private fun resizeBilinear(image: RgbImage, targetWidth: Int, targetHeight: Int): ByteArray {
        val xMap = axisMap(image.width, targetWidth)
        val yMap = axisMap(image.height, targetHeight)
        val denominator = xMap[0].denominator * yMap[0].denominator
        val result = ByteArray(targetWidth * targetHeight * 3)
        for (y in 0 until targetHeight) {
            val ym = yMap[y]
            val rowLow = ym.low * image.width
            val rowHigh = ym.high * image.width
            for (x in 0 until targetWidth) {
                val xm = xMap[x]
                val base00 = (rowLow + xm.low) * 3
                val base01 = (rowLow + xm.high) * 3
                val base10 = (rowHigh + xm.low) * 3
                val base11 = (rowHigh + xm.high) * 3
                val weight00 = xm.weightLow * ym.weightLow
                val weight01 = xm.weightHigh * ym.weightLow
                val weight10 = xm.weightLow * ym.weightHigh
                val weight11 = xm.weightHigh * ym.weightHigh
                val outputBase = (y * targetWidth + x) * 3
                for (channel in 0..2) {
                    val numerator =
                        (image.rgb[base00 + channel].toInt() and 0xff).toLong() * weight00 +
                        (image.rgb[base01 + channel].toInt() and 0xff).toLong() * weight01 +
                        (image.rgb[base10 + channel].toInt() and 0xff).toLong() * weight10 +
                        (image.rgb[base11 + channel].toInt() and 0xff).toLong() * weight11
                    val value = (numerator + denominator / 2L) / denominator
                    result[outputBase + channel] = value.toByte()
                }
            }
        }
        return result
    }
}
