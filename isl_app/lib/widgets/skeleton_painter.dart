import 'package:flutter/material.dart';
import '../services/text_to_sign_service.dart';

/// Custom painter to draw 2D skeleton from landmarks
class SkeletonPainter extends CustomPainter {
  final SignFrame? frame;
  final String signType; // "word", "letter", "number"
  final Color poseColor;
  final Color leftHandColor;
  final Color rightHandColor;

  SkeletonPainter({
    this.frame,
    this.signType = 'word',
    this.poseColor = Colors.white,
    this.leftHandColor = const Color(0xFF4CAF50),
    this.rightHandColor = const Color(0xFF2196F3),
  });

  // Pose connections (MediaPipe indices)
  static const List<List<int>> poseConnections = [
    // Torso
    [11, 12], // shoulders
    [11, 23], [12, 24], // shoulders to hips
    [23, 24], // hips
    // Left arm
    [11, 13], [13, 15], // shoulder -> elbow -> wrist
    // Right arm
    [12, 14], [14, 16], // shoulder -> elbow -> wrist
    // Face (simplified)
    [0, 1], [1, 2], [2, 3], [3, 7], // left eyebrow to ear
    [0, 4], [4, 5], [5, 6], [6, 8], // right eyebrow to ear
    [9, 10], // mouth
  ];

  // Hand connections (MediaPipe indices)
  static const List<List<int>> handConnections = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle finger
    [0, 9], [9, 10], [10, 11], [11, 12],
    // Ring finger
    [0, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm
    [5, 9], [9, 13], [13, 17],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    if (frame == null) {
      _drawPlaceholder(canvas, size);
      return;
    }

    bool hasAnyData = false;
    bool isFingerSpelling = signType == 'letter' || signType == 'number';

    // Draw pose only for word signs (skip for letters/numbers to avoid jumping)
    if (!isFingerSpelling && frame!.pose != null && frame!.pose!.isNotEmpty) {
      hasAnyData = true;
      _drawSkeleton(
        canvas, size,
        frame!.pose!,
        poseConnections,
        poseColor,
        jointRadius: 4,
        boneWidth: 2,
      );
    }

    // Draw left hand
    if (frame!.leftHand != null && frame!.leftHand!.isNotEmpty) {
      hasAnyData = true;
      if (!isFingerSpelling && frame!.pose != null && frame!.pose!.length > 15) {
        // Position relative to wrist from pose (for words)
        _drawHand(
          canvas, size,
          frame!.leftHand!,
          frame!.pose![15],
          leftHandColor,
        );
      } else {
        // Center hand for fingerspelling
        _drawHandCentered(
          canvas, size,
          frame!.leftHand!,
          leftHandColor,
          offsetX: frame!.rightHand != null ? -0.2 : 0.0,
        );
      }
    }

    // Draw right hand
    if (frame!.rightHand != null && frame!.rightHand!.isNotEmpty) {
      hasAnyData = true;
      if (!isFingerSpelling && frame!.pose != null && frame!.pose!.length > 16) {
        // Position relative to wrist from pose (for words)
        _drawHand(
          canvas, size,
          frame!.rightHand!,
          frame!.pose![16],
          rightHandColor,
        );
      } else {
        // No pose - draw hand centered (for letter/number signs)
        _drawHandCentered(
          canvas, size,
          frame!.rightHand!,
          rightHandColor,
          offsetX: 0.15, // Slightly right of center
        );
      }
    }

    // If no data at all, show placeholder
    if (!hasAnyData) {
      _drawPlaceholder(canvas, size);
    }
  }

  void _drawSkeleton(
    Canvas canvas,
    Size size,
    List<List<double>> landmarks,
    List<List<int>> connections,
    Color color, {
    double jointRadius = 3,
    double boneWidth = 2,
  }) {
    final jointPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final bonePaint = Paint()
      ..color = color.withOpacity(0.7)
      ..strokeWidth = boneWidth
      ..style = PaintingStyle.stroke;

    // Draw bones
    for (var connection in connections) {
      if (connection[0] < landmarks.length && connection[1] < landmarks.length) {
        var p1 = landmarks[connection[0]];
        var p2 = landmarks[connection[1]];

        canvas.drawLine(
          Offset(p1[0] * size.width, p1[1] * size.height),
          Offset(p2[0] * size.width, p2[1] * size.height),
          bonePaint,
        );
      }
    }

    // Draw joints
    for (var landmark in landmarks) {
      canvas.drawCircle(
        Offset(landmark[0] * size.width, landmark[1] * size.height),
        jointRadius,
        jointPaint,
      );
    }
  }

  void _drawHand(
    Canvas canvas,
    Size size,
    List<List<double>> handLandmarks,
    List<double> wristPos,
    Color color,
  ) {
    final jointPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final bonePaint = Paint()
      ..color = color.withOpacity(0.8)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    // Scale factor for hand (hands are in normalized 0-1 space)
    double scale = 0.15;

    // Draw bones
    for (var connection in handConnections) {
      if (connection[0] < handLandmarks.length && connection[1] < handLandmarks.length) {
        var p1 = handLandmarks[connection[0]];
        var p2 = handLandmarks[connection[1]];

        // Position relative to wrist
        double x1 = (wristPos[0] + (p1[0] - 0.5) * scale) * size.width;
        double y1 = (wristPos[1] + (p1[1] - 0.5) * scale) * size.height;
        double x2 = (wristPos[0] + (p2[0] - 0.5) * scale) * size.width;
        double y2 = (wristPos[1] + (p2[1] - 0.5) * scale) * size.height;

        canvas.drawLine(Offset(x1, y1), Offset(x2, y2), bonePaint);
      }
    }

    // Draw joints
    for (var landmark in handLandmarks) {
      double x = (wristPos[0] + (landmark[0] - 0.5) * scale) * size.width;
      double y = (wristPos[1] + (landmark[1] - 0.5) * scale) * size.height;
      canvas.drawCircle(Offset(x, y), 2, jointPaint);
    }
  }

  /// Draw hand centered on screen (for letter/number signs without pose data)
  void _drawHandCentered(
    Canvas canvas,
    Size size,
    List<List<double>> handLandmarks,
    Color color, {
    double offsetX = 0.0,
  }) {
    final jointPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final bonePaint = Paint()
      ..color = color.withOpacity(0.9)
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    // Calculate bounding box of hand landmarks
    double minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
    for (var lm in handLandmarks) {
      if (lm[0] < minX) minX = lm[0];
      if (lm[0] > maxX) maxX = lm[0];
      if (lm[1] < minY) minY = lm[1];
      if (lm[1] > maxY) maxY = lm[1];
    }

    // Center and scale to fit nicely in the view
    double handWidth = maxX - minX;
    double handHeight = maxY - minY;
    double handCenterX = (minX + maxX) / 2;
    double handCenterY = (minY + maxY) / 2;

    // Scale to fill about 60% of the smaller dimension
    double scale = 0.6 / (handWidth > handHeight ? handWidth : handHeight);
    scale = scale.clamp(0.8, 2.5); // Reasonable bounds

    // Target center position
    double targetX = 0.5 + offsetX;
    double targetY = 0.45; // Slightly above center

    // Draw bones
    for (var connection in handConnections) {
      if (connection[0] < handLandmarks.length && connection[1] < handLandmarks.length) {
        var p1 = handLandmarks[connection[0]];
        var p2 = handLandmarks[connection[1]];

        double x1 = (targetX + (p1[0] - handCenterX) * scale) * size.width;
        double y1 = (targetY + (p1[1] - handCenterY) * scale) * size.height;
        double x2 = (targetX + (p2[0] - handCenterX) * scale) * size.width;
        double y2 = (targetY + (p2[1] - handCenterY) * scale) * size.height;

        canvas.drawLine(Offset(x1, y1), Offset(x2, y2), bonePaint);
      }
    }

    // Draw joints with gradient-like effect (larger at fingertips)
    for (int i = 0; i < handLandmarks.length; i++) {
      var landmark = handLandmarks[i];
      double x = (targetX + (landmark[0] - handCenterX) * scale) * size.width;
      double y = (targetY + (landmark[1] - handCenterY) * scale) * size.height;
      
      // Fingertips (4, 8, 12, 16, 20) are larger
      double radius = [4, 8, 12, 16, 20].contains(i) ? 5.0 : 3.0;
      canvas.drawCircle(Offset(x, y), radius, jointPaint);
    }
  }

  void _drawPlaceholder(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey.withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    // Draw simple stick figure placeholder
    double cx = size.width / 2;
    double cy = size.height / 3;

    // Head
    canvas.drawCircle(Offset(cx, cy), 20, paint);

    // Body
    canvas.drawLine(Offset(cx, cy + 20), Offset(cx, cy + 80), paint);

    // Arms
    canvas.drawLine(Offset(cx - 40, cy + 40), Offset(cx + 40, cy + 40), paint);

    // Legs
    canvas.drawLine(Offset(cx, cy + 80), Offset(cx - 30, cy + 130), paint);
    canvas.drawLine(Offset(cx, cy + 80), Offset(cx + 30, cy + 130), paint);
  }

  @override
  bool shouldRepaint(covariant SkeletonPainter oldDelegate) {
    return frame != oldDelegate.frame || signType != oldDelegate.signType;
  }
}

/// Widget to display the skeleton animation
class SkeletonAvatar extends StatelessWidget {
  final SignFrame? frame;
  final String signType;
  final Color backgroundColor;

  const SkeletonAvatar({
    super.key,
    this.frame,
    this.signType = 'word',
    this.backgroundColor = const Color(0xFF1A1A2E),
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(16),
      ),
      child: CustomPaint(
        painter: SkeletonPainter(frame: frame, signType: signType),
        size: Size.infinite,
      ),
    );
  }
}
