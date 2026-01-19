plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.isl_translator"
    compileSdk = 36
    buildToolsVersion = "35.0.0"
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        applicationId = "com.example.isl_translator"
        minSdk = 24  // MediaPipe requires API 24+
        targetSdk = 35
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    
    // Required for MediaPipe native libraries
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // MediaPipe Tasks Vision - for Pose and Hand Landmarker
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}

flutter {
    source = "../.."
}
