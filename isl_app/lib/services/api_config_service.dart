import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;

/// Service to manage backend API configuration
/// Allows dynamic IP address configuration without recompiling
class ApiConfigService {
  static final ApiConfigService _instance = ApiConfigService._internal();
  factory ApiConfigService() => _instance;
  ApiConfigService._internal();

  static const String _ipKey = 'backend_ip';
  static const String _portKey = 'backend_port';
  static const String _defaultPort = '8000';

  String? _cachedIp;
  String _cachedPort = _defaultPort;
  bool _isConnected = false;
  bool _isInitialized = false;

  // Stream for connection status updates
  final _connectionController = StreamController<bool>.broadcast();
  Stream<bool> get connectionStream => _connectionController.stream;
  
  bool get isConnected => _isConnected;
  bool get isInitialized => _isInitialized;
  String? get currentIp => _cachedIp;
  String get currentPort => _cachedPort;

  String? get baseUrl {
    if (_cachedIp == null || _cachedIp!.isEmpty) return null;
    return 'http://$_cachedIp:$_cachedPort';
  }

  /// Initialize and load saved settings
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    try {
      final prefs = await SharedPreferences.getInstance();
      _cachedIp = prefs.getString(_ipKey);
      _cachedPort = prefs.getString(_portKey) ?? _defaultPort;
      _isInitialized = true;

      if (kDebugMode) {
        debugPrint('ApiConfig: Loaded IP=$_cachedIp, Port=$_cachedPort');
      }

      if (_cachedIp != null && _cachedIp!.isNotEmpty) {
        // Check connection on startup (don't await - do in background)
        checkConnection();
      }
    } catch (e) {
      if (kDebugMode) debugPrint('ApiConfig: Init error - $e');
      _isInitialized = true;
    }
  }

  /// Save IP and port settings
  Future<void> saveSettings(String ip, String port) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_ipKey, ip.trim());
      await prefs.setString(_portKey, port.trim());
      _cachedIp = ip.trim();
      _cachedPort = port.trim();
      
      if (kDebugMode) {
        debugPrint('ApiConfig: Saved IP=$_cachedIp, Port=$_cachedPort');
      }
      
      // Check connection after saving
      await checkConnection();
    } catch (e) {
      if (kDebugMode) debugPrint('ApiConfig: Save error - $e');
    }
  }

  /// Check if backend is reachable
  Future<bool> checkConnection() async {
    if (_cachedIp == null || _cachedIp!.isEmpty) {
      _isConnected = false;
      _connectionController.add(false);
      return false;
    }

    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 3));

      _isConnected = response.statusCode == 200;
      _connectionController.add(_isConnected);
      
      if (kDebugMode) {
        debugPrint('ApiConfig: Connection check - ${_isConnected ? "OK" : "Failed"}');
      }
      
      return _isConnected;
    } catch (e) {
      if (kDebugMode) debugPrint('ApiConfig: Connection error - $e');
      _isConnected = false;
      _connectionController.add(false);
      return false;
    }
  }

  /// Auto-discover backend on local network
  /// Scans common IPs on the same subnet
  Future<String?> autoDiscover({
    void Function(String status)? onProgress,
  }) async {
    onProgress?.call('Getting local IP...');
    
    // Get device's local IP to determine subnet
    String? localSubnet;
    try {
      final interfaces = await NetworkInterface.list();
      for (var interface in interfaces) {
        for (var addr in interface.addresses) {
          if (addr.type == InternetAddressType.IPv4 && !addr.isLoopback) {
            final parts = addr.address.split('.');
            if (parts.length == 4) {
              localSubnet = '${parts[0]}.${parts[1]}.${parts[2]}';
              if (kDebugMode) {
                debugPrint('ApiConfig: Local subnet = $localSubnet');
              }
              break;
            }
          }
        }
        if (localSubnet != null) break;
      }
    } catch (e) {
      if (kDebugMode) debugPrint('ApiConfig: Failed to get local IP - $e');
      return null;
    }

    if (localSubnet == null) {
      onProgress?.call('Could not determine network');
      return null;
    }

    onProgress?.call('Scanning network $localSubnet.x ...');

    // Common IP addresses to check first (routers, DHCP assigned, etc.)
    final priorityLastOctets = [1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                 2, 50, 51, 52, 150, 151, 200, 201, 10, 11, 20, 21];

    for (var lastOctet in priorityLastOctets) {
      final ip = '$localSubnet.$lastOctet';
      onProgress?.call('Checking $ip...');
      
      try {
        final response = await http
            .get(Uri.parse('http://$ip:$_cachedPort/health'))
            .timeout(const Duration(milliseconds: 500));

        if (response.statusCode == 200) {
          if (kDebugMode) debugPrint('ApiConfig: Found backend at $ip');
          await saveSettings(ip, _cachedPort);
          onProgress?.call('Found backend at $ip!');
          return ip;
        }
      } catch (_) {
        // Continue to next IP
      }
    }

    onProgress?.call('Backend not found on network');
    return null;
  }

  /// Get the device's local IP address
  Future<String?> getLocalIp() async {
    try {
      final interfaces = await NetworkInterface.list();
      for (var interface in interfaces) {
        for (var addr in interface.addresses) {
          if (addr.type == InternetAddressType.IPv4 && !addr.isLoopback) {
            return addr.address;
          }
        }
      }
    } catch (_) {}
    return null;
  }

  /// Clear saved settings
  Future<void> clearSettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_ipKey);
      await prefs.remove(_portKey);
      _cachedIp = null;
      _cachedPort = _defaultPort;
      _isConnected = false;
      _connectionController.add(false);
    } catch (e) {
      if (kDebugMode) debugPrint('ApiConfig: Clear error - $e');
    }
  }

  void dispose() {
    _connectionController.close();
  }
}
