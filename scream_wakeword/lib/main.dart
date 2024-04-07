import 'package:flutter/material.dart';
import 'package:scream_wakeword/constants/route.dart';
import 'package:scream_wakeword/views/connection.dart';
import 'package:scream_wakeword/views/recorder.dart';

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
      home: const ConnectionView(),
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
