package com.bedjamahdi.scanpestai.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloPostProcessorTest {

    private val labels = listOf("aphids", "grub", "wireworm")
    private val numChannel = 4 + labels.size

    /** Builds a YOLO output tensor laid out as [channel][candidate], flattened. */
    private fun output(vararg candidates: FloatArray): FloatArray {
        val numElements = candidates.size
        val array = FloatArray(numChannel * numElements)
        candidates.forEachIndexed { c, values ->
            values.forEachIndexed { ch, v -> array[ch * numElements + c] = v }
        }
        return array
    }

    private fun candidate(cx: Float, cy: Float, w: Float, h: Float, vararg scores: Float) =
        floatArrayOf(cx, cy, w, h, *scores)

    private fun box(x1: Float, y1: Float, x2: Float, y2: Float, cnf: Float, cls: Int = 0) =
        BoundingBox(
            x1 = x1, y1 = y1, x2 = x2, y2 = y2,
            cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, w = x2 - x1, h = y2 - y1,
            cnf = cnf, cls = cls, clsName = labels[cls]
        )

    @Test
    fun decode_picksHighestScoringClassAndConvertsToCorners() {
        val array = output(candidate(0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.9f, 0.3f))

        val boxes = YoloPostProcessor.decode(array, numChannel, 1, labels, 0.3f)

        assertEquals(1, boxes.size)
        val b = boxes[0]
        assertEquals(1, b.cls)
        assertEquals("grub", b.clsName)
        assertEquals(0.9f, b.cnf, 1e-6f)
        assertEquals(0.4f, b.x1, 1e-6f)
        assertEquals(0.3f, b.y1, 1e-6f)
        assertEquals(0.6f, b.x2, 1e-6f)
        assertEquals(0.7f, b.y2, 1e-6f)
    }

    @Test
    fun decode_dropsCandidatesBelowConfidenceThreshold() {
        val array = output(
            candidate(0.5f, 0.5f, 0.2f, 0.2f, 0.2f, 0.1f, 0.25f),
            candidate(0.3f, 0.3f, 0.2f, 0.2f, 0.8f, 0.1f, 0.1f)
        )

        val boxes = YoloPostProcessor.decode(array, numChannel, 2, labels, 0.3f)

        assertEquals(listOf("aphids"), boxes.map { it.clsName })
    }

    @Test
    fun decode_dropsBoxesExtendingOutsideTheImage() {
        val array = output(
            candidate(0.05f, 0.5f, 0.2f, 0.2f, 0.9f, 0f, 0f), // x1 < 0
            candidate(0.5f, 0.95f, 0.2f, 0.2f, 0.9f, 0f, 0f), // y2 > 1
            candidate(0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0f, 0f)   // fully inside
        )

        val boxes = YoloPostProcessor.decode(array, numChannel, 3, labels, 0.3f)

        assertEquals(1, boxes.size)
        assertEquals(0.5f, boxes[0].cx, 1e-6f)
    }

    @Test
    fun decode_returnsEmptyListWhenNothingDetected() {
        val array = output(candidate(0.5f, 0.5f, 0.2f, 0.2f, 0.01f, 0.02f, 0.03f))

        assertTrue(YoloPostProcessor.decode(array, numChannel, 1, labels, 0.3f).isEmpty())
    }

    @Test
    fun iou_identicalBoxesIsOne() {
        val a = box(0.1f, 0.1f, 0.5f, 0.5f, 0.9f)
        assertEquals(1f, YoloPostProcessor.calculateIoU(a, a), 1e-6f)
    }

    @Test
    fun iou_disjointBoxesIsZero() {
        val a = box(0.0f, 0.0f, 0.2f, 0.2f, 0.9f)
        val b = box(0.5f, 0.5f, 0.7f, 0.7f, 0.9f)
        assertEquals(0f, YoloPostProcessor.calculateIoU(a, b), 1e-6f)
    }

    @Test
    fun iou_partialOverlap() {
        // Two 0.2x0.2 boxes overlapping on a 0.1x0.2 strip: 0.02 / (0.04 + 0.04 - 0.02) = 1/3
        val a = box(0.0f, 0.0f, 0.2f, 0.2f, 0.9f)
        val b = box(0.1f, 0.0f, 0.3f, 0.2f, 0.9f)
        assertEquals(1f / 3f, YoloPostProcessor.calculateIoU(a, b), 1e-5f)
    }

    @Test
    fun nms_keepsHighestConfidenceAmongOverlappingBoxes() {
        val best = box(0.10f, 0.10f, 0.50f, 0.50f, 0.95f)
        val duplicate = box(0.12f, 0.12f, 0.52f, 0.52f, 0.80f)
        val separate = box(0.60f, 0.60f, 0.90f, 0.90f, 0.70f, cls = 2)

        val kept = YoloPostProcessor.applyNMS(listOf(duplicate, separate, best), 0.5f)

        assertEquals(listOf(best, separate), kept)
    }

    @Test
    fun nms_keepsBoxesBelowIouThreshold() {
        val a = box(0.0f, 0.0f, 0.2f, 0.2f, 0.9f)
        val b = box(0.1f, 0.0f, 0.3f, 0.2f, 0.8f) // IoU = 1/3

        assertEquals(2, YoloPostProcessor.applyNMS(listOf(a, b), 0.5f).size)
    }

    @Test
    fun nms_emptyInput() {
        assertTrue(YoloPostProcessor.applyNMS(emptyList(), 0.5f).isEmpty())
    }
}
