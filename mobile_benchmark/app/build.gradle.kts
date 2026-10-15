plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.lmdyolo.benchmark"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.lmdyolo.benchmark"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        buildConfigField("String", "ORT_VERSION", "\"1.24.3\"")
    }

    buildFeatures {
        buildConfig = true
    }

    androidResources {
        noCompress += listOf("onnx", "bin")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
}
