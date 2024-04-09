import 'dart:async';
import 'dart:io';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;

class RecorderView extends StatefulWidget {
  const RecorderView({super.key});

  @override
  State<RecorderView> createState() => _RecorderViewState();
}

class _RecorderViewState extends State<RecorderView> {
  final recorder = FlutterSoundRecorder();
  // String customDirectoryPath = '/storage/emulated/0/Recordings';
  bool isRecorderReady = false;

  @override
  void initState() {
    print("I am in init");
    super.initState();

    initRecorder();
  }

  @override
  void dispose() {
    recorder.closeRecorder();

    super.dispose();
  }

  Future initRecorder() async {
    // print("I am here");
    debugPrint(" I am here");
    final status = await Permission.microphone.request();

    if (status != PermissionStatus.granted) {
      throw 'Microphone permission not granted';
    }

    await recorder.openRecorder();
    isRecorderReady = true;

    recorder.setSubscriptionDuration(
      const Duration(milliseconds: 500),
    );
  }

  Future record() async {
    if (!isRecorderReady) return;
    await recorder.startRecorder(toFile: 'audio');
  }

  Future stop() async {
    if (!isRecorderReady) return;

    final path = await recorder.stopRecorder();
    final audioFile = File(path!);

    print('Recorded audio: $audioFile');
    _showOptionsDialog(path);
  }

  Future<void> _showOptionsDialog(path) async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Recording Options'),
          content: SingleChildScrollView(
            child: ListBody(
              children: <Widget>[
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                    retakeRecording(); // Implement retake action
                  },
                  child: Text('Retake'),
                ),
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                    sendRecordingToServer(path); // Implement send action
                  },
                  child: Text('Send'),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
  Future<void> retakeRecording() async {
    // Implement retake logic here
  }

  Future<void> sendRecordingToServer(String filePath) async {
    final url = Uri.parse('http://192.168.1.13:5000/upload-audio');
    final file = File(filePath);

    // Check if the file exists
    if (!await file.exists()) {
      print('File not found: $filePath');
      return;
    }

    // Read the file content as bytes
    List<int> fileBytes = await file.readAsBytes();

    // Send the file content to the server
    final response = await http.post(
      url,
      body: fileBytes,
      headers: <String, String>{
        'Content-Type': 'application/octet-stream',
      },
    );

    if (response.statusCode == 200) {
      print('Audio file sent successfully');
    } else {
      print('Failed to send audio file. Error code: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            StreamBuilder<RecordingDisposition>(
              stream: recorder.onProgress,
              builder: (context, snapshot) {
                final duration =
                    snapshot.hasData ? snapshot.data!.duration : Duration.zero;
                String twoDigits(int n) => n.toString().padLeft(2, '0');
                final twoDigitMinutes =
                    twoDigits(duration.inMinutes.remainder(60));
                final twoDigitSeconds =
                    twoDigits(duration.inSeconds.remainder(60));

                return Text(
                  '$twoDigitMinutes:$twoDigitSeconds',
                  style: const TextStyle(
                    fontSize: 70,
                    fontWeight: FontWeight.bold,
                  ),
                );
              },
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              child: Icon(
                recorder.isRecording ? Icons.stop : Icons.mic,
                size: 70,
              ),
              onPressed: () async {
                if (recorder.isRecording) {
                  await stop();
                } else {
                  await record();
                }
                setState(() {});
                // !recorder.isRecording;
              },
            ),
          ],
        ),
      ),
    );
  }
}
