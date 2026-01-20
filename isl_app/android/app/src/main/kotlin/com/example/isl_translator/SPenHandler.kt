package com.example.isl_translator

import android.content.Context
import android.util.Log
import android.view.KeyEvent
import io.flutter.plugin.common.EventChannel

/**
 * S Pen Air Actions Handler for Samsung Galaxy Tab S9
 * 
 * Detects S Pen button clicks and air gestures and sends them to Flutter.
 * Works by intercepting KeyEvents from the S Pen remote.
 */
class SPenHandler(private val context: Context) : EventChannel.StreamHandler {
    
    companion object {
        private const val TAG = "SPenHandler"
        
        // S Pen button KeyCodes (Samsung specific)
        const val SPEN_BUTTON_CLICK = KeyEvent.KEYCODE_STYLUS_BUTTON_PRIMARY  // Single click
        const val SPEN_BUTTON_DOUBLE = KeyEvent.KEYCODE_STYLUS_BUTTON_SECONDARY // Double click
        
        // Air Action gesture types (simplified - actual gestures come via different mechanism)
        const val GESTURE_SWIPE_LEFT = "swipe_left"
        const val GESTURE_SWIPE_RIGHT = "swipe_right"
        const val GESTURE_SWIPE_UP = "swipe_up"
        const val GESTURE_SWIPE_DOWN = "swipe_down"
        const val GESTURE_CIRCLE_CW = "circle_cw"
        const val GESTURE_CIRCLE_CCW = "circle_ccw"
        const val GESTURE_BUTTON_CLICK = "button_click"
        const val GESTURE_BUTTON_LONG = "button_long"
    }
    
    private var eventSink: EventChannel.EventSink? = null
    private var lastClickTime: Long = 0
    private var clickCount: Int = 0
    
    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
        Log.d(TAG, "S Pen event channel listening")
    }
    
    override fun onCancel(arguments: Any?) {
        eventSink = null
        Log.d(TAG, "S Pen event channel cancelled")
    }
    
    /**
     * Handle KeyEvent from MainActivity
     * Returns true if the event was consumed
     */
    fun handleKeyEvent(keyCode: Int, event: KeyEvent): Boolean {
        Log.d(TAG, "KeyEvent received: keyCode=$keyCode, action=${event.action}")
        
        when (keyCode) {
            // S Pen button primary (single click on pen button)
            KeyEvent.KEYCODE_STYLUS_BUTTON_PRIMARY,
            287 -> {  // KEYCODE_BUTTON_STYLUS1 = 287
                if (event.action == KeyEvent.ACTION_DOWN) {
                    handleButtonPress()
                    return true
                } else if (event.action == KeyEvent.ACTION_UP) {
                    handleButtonRelease(event.eventTime - event.downTime)
                    return true
                }
            }
            
            // S Pen button secondary (usually double click)
            KeyEvent.KEYCODE_STYLUS_BUTTON_SECONDARY,
            288 -> {  // KEYCODE_BUTTON_STYLUS2 = 288
                if (event.action == KeyEvent.ACTION_UP) {
                    sendGesture(GESTURE_BUTTON_CLICK)
                    return true
                }
            }
            
            // Volume buttons can also be used as backup controls
            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    sendGesture(GESTURE_SWIPE_UP)
                    return true
                }
            }
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    sendGesture(GESTURE_SWIPE_DOWN)
                    return true
                }
            }
        }
        
        return false
    }
    
    private fun handleButtonPress() {
        // Track for long press detection
    }
    
    private fun handleButtonRelease(pressDuration: Long) {
        if (pressDuration > 500) {
            // Long press
            sendGesture(GESTURE_BUTTON_LONG)
        } else {
            // Short click - check for double click
            val now = System.currentTimeMillis()
            if (now - lastClickTime < 300) {
                clickCount++
                if (clickCount >= 2) {
                    // Double click detected
                    sendGesture("button_double")
                    clickCount = 0
                }
            } else {
                clickCount = 1
                sendGesture(GESTURE_BUTTON_CLICK)
            }
            lastClickTime = now
        }
    }
    
    /**
     * Send a gesture event to Flutter
     */
    fun sendGesture(gesture: String) {
        Log.d(TAG, "Sending gesture to Flutter: $gesture")
        eventSink?.success(mapOf(
            "type" to "gesture",
            "gesture" to gesture,
            "timestamp" to System.currentTimeMillis()
        ))
    }
    
    /**
     * Send a demo output directly (for presentation purposes)
     */
    fun sendDemoOutput(text: String) {
        Log.d(TAG, "Sending demo output: $text")
        eventSink?.success(mapOf(
            "type" to "demo_output",
            "text" to text,
            "timestamp" to System.currentTimeMillis()
        ))
    }
}
