import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// On-device screens
import 'screens/sign_to_text_screen.dart';
import 'screens/text_to_sign_screen.dart';  // Uses TextToSignService with fingerspelling
import 'services/api_config_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ISLTranslatorApp());
}

class ISLTranslatorApp extends StatelessWidget {
  const ISLTranslatorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ISL Translator',
      debugShowCheckedModeBanner: false,
      theme: _buildTheme(),
      home: const MainNavigationScreen(),
    );
  }

  ThemeData _buildTheme() {
    // Clean, minimal theme with neutral colors
    const primaryColor = Color(0xFF1A1A2E); // Dark navy
    const accentColor = Color(0xFF16213E); // Slightly lighter navy
    const surfaceColor = Color(0xFFF5F5F7); // Light gray
    const textColor = Color(0xFF2D2D2D); // Dark gray text

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,

      // Color scheme - neutral, professional
      colorScheme: const ColorScheme.light(
        primary: primaryColor,
        secondary: accentColor,
        surface: surfaceColor,
        surfaceContainerHighest: Colors.white,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: textColor,
        onSurfaceVariant: textColor,
      ),

      // App bar - clean and minimal
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.white,
        foregroundColor: textColor,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.inter(
          fontSize: 18,
          fontWeight: FontWeight.w600,
          color: textColor,
        ),
      ),

      // Text theme
      textTheme: GoogleFonts.interTextTheme().copyWith(
        headlineLarge: GoogleFonts.inter(
          fontSize: 28,
          fontWeight: FontWeight.w700,
          color: textColor,
        ),
        headlineMedium: GoogleFonts.inter(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: textColor,
        ),
        bodyLarge: GoogleFonts.inter(
          fontSize: 16,
          fontWeight: FontWeight.w400,
          color: textColor,
        ),
        bodyMedium: GoogleFonts.inter(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: textColor,
        ),
        labelLarge: GoogleFonts.inter(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: textColor,
        ),
      ),

      // Elevated button - rounded, minimal
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: GoogleFonts.inter(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),

      // Card theme
      cardTheme: CardThemeData(
        color: Colors.white,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: Colors.grey.shade200),
        ),
      ),

      // Input decoration
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceColor,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),

      // Scaffold background
      scaffoldBackgroundColor: Colors.white,
    );
  }
}

/// Main navigation with bottom bar for dual-mode translation
class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;
  final ApiConfigService _apiConfig = ApiConfigService();
  bool _isConnected = false;

  final List<Widget> _screens = [
    const SignToTextScreen(), // Sign → Text (camera + on-device ML)
    const TextToSignScreen(), // Text → Sign (offline stick figure with fingerspelling)
  ];

  @override
  void initState() {
    super.initState();
    _initializeApi();
  }

  Future<void> _initializeApi() async {
    await _apiConfig.initialize();
    final connected = await _apiConfig.checkConnection();
    if (mounted) {
      setState(() => _isConnected = connected);
    }
  }

  void _showApiConfigDialog() {
    final ipController = TextEditingController(text: _apiConfig.currentIp ?? '');
    final portController = TextEditingController(text: _apiConfig.currentPort);
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Backend Server'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Icon(
                  _isConnected ? Icons.check_circle : Icons.error,
                  color: _isConnected ? Colors.green : Colors.red,
                  size: 20,
                ),
                const SizedBox(width: 8),
                Text(
                  _isConnected ? 'Connected' : 'Not Connected',
                  style: TextStyle(
                    color: _isConnected ? Colors.green : Colors.red,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextField(
              controller: ipController,
              decoration: const InputDecoration(
                labelText: 'IP Address',
                hintText: 'e.g., 192.168.1.100',
                prefixIcon: Icon(Icons.computer),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: portController,
              decoration: const InputDecoration(
                labelText: 'Port',
                hintText: '8000',
                prefixIcon: Icon(Icons.numbers),
              ),
              keyboardType: TextInputType.number,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              // Try auto-discovery
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Scanning network...')),
              );
              final ip = await _apiConfig.autoDiscover();
              if (ip != null) {
                setState(() => _isConnected = true);
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Found backend at $ip')),
                  );
                }
              } else {
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Backend not found on network')),
                  );
                }
              }
            },
            child: const Text('Auto-Find'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              final ip = ipController.text.trim();
              final port = portController.text.trim();
              if (ip.isNotEmpty) {
                await _apiConfig.saveSettings(ip, port.isEmpty ? '8000' : port);
                final connected = await _apiConfig.checkConnection();
                setState(() => _isConnected = connected);
                if (mounted) {
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(connected 
                        ? 'Connected to $ip:$port' 
                        : 'Failed to connect to $ip:$port'),
                    ),
                  );
                }
              }
            },
            child: const Text('Connect'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ISL Translator'),
        actions: [
          // Connection status indicator + settings button
          IconButton(
            onPressed: _showApiConfigDialog,
            icon: Stack(
              children: [
                const Icon(Icons.settings),
                Positioned(
                  right: 0,
                  top: 0,
                  child: Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      color: _isConnected ? Colors.green : Colors.red,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ],
            ),
            tooltip: 'Server Settings',
          ),
        ],
      ),
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) => setState(() => _currentIndex = index),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.videocam_outlined),
            selectedIcon: Icon(Icons.videocam),
            label: 'Sign → Text',
          ),
          NavigationDestination(
            icon: Icon(Icons.accessibility_new_outlined),
            selectedIcon: Icon(Icons.accessibility_new),
            label: 'Text → Sign',
          ),
        ],
      ),
    );
  }
}
