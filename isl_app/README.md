# ISL Translator - Flutter App

A clean, minimal mobile app for Indian Sign Language to English translation.

## Features

- Real-time camera preview
- Hold-to-record video capture
- Translation via API backend
- Translation history
- Configurable API endpoint

## UI Design

- Clean, minimal interface
- Neutral color palette (dark navy + grays)
- No bright colors or emojis
- Professional appearance

## Setup

### 1. Install Flutter
```bash
# Download from https://flutter.dev/docs/get-started/install
# Or via chocolatey on Windows:
choco install flutter
```

### 2. Get Dependencies
```bash
cd isl_app
flutter pub get
```

### 3. Run on Device/Emulator
```bash
flutter run
```

### 4. Build for Production
```bash
# Android
flutter build apk --release

# iOS
flutter build ios --release
```

## Configuration

The app connects to `http://localhost:8000` by default.

To change the API endpoint:
1. Tap the settings icon in the top right
2. Enter your API URL
3. Save

For production, update `TranslationService.baseUrl` in:
`lib/services/translation_service.dart`

## Usage

1. Point camera at signer
2. **Hold** the record button while signing
3. **Release** to translate
4. View translation below camera

## Project Structure

```
isl_app/
├── lib/
│   ├── main.dart              # App entry + theme
│   ├── screens/
│   │   └── home_screen.dart   # Main camera + translation UI
│   └── services/
│       └── translation_service.dart  # API client
├── pubspec.yaml               # Dependencies
└── README.md
```

## Required Permissions

- Camera (for video recording)
- Internet (for API calls)
