import 'package:flutter/material.dart';
import 'package:recorder_frontend/constants/route.dart';
import 'package:recorder_frontend/views/connection.dart';
import 'package:recorder_frontend/views/recorder.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.purple,
      ),
      themeMode: ThemeMode.dark,
      home: const RecorderView(),
      routes: {
        // messageRoute: (context) => const MessageView(),
        // profileRoute: (context) => const ProfileView(),
        connectionRoute: (context) => const ConnectionView(),
        recorderRoute: (context) => const RecorderView(),
        // chatRoute: (context) => ChatView(),
        // chatUIRoute: (context) => const ChatUserInterfaceView(),
      },
    ),
  );
}
