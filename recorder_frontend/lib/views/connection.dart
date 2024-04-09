import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class ConnectionView extends StatefulWidget {
  const ConnectionView({Key? key});

  @override
  State<ConnectionView> createState() => _ConnectionState();
}

class _ConnectionState extends State<ConnectionView> {
  final TextEditingController _nameController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Connection'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(
                  labelText: 'Name',
                ),
              ),
            ),
            ElevatedButton(
              onPressed: () {
                sendDataToServer(_nameController.text);
              },
              child: const Text('Send Data to Server'),
            ),
          ],
        ),
      ),
    );
  }

  void sendDataToServer(String data) async {
    final url = Uri.parse('http://192.168.1.13:5000/data');
    final response = await http.post(
      url,
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{
        'text': data,
      }),
    );

    if (response.statusCode == 200) {
      print('Data sent successfully');
    } else {
      print('Failed to send data. Error code: ${response.statusCode}');
    }
  }
}

// import 'dart:convert';
// import 'dart:io';

// import 'package:flutter/material.dart';

// // import 'audio_recording_page.dart'; // Assuming your audio recording page is in a separate file

// class ConnectionView extends StatefulWidget {
//   const ConnectionView({Key? key});

//   @override
//   State<ConnectionView> createState() => _ConnectionState();
// }

// class _ConnectionState extends State<ConnectionView> {
//   final TextEditingController _iPController = TextEditingController();
//   final TextEditingController _nameController = TextEditingController();

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         backgroundColor: Theme.of(context).colorScheme.inversePrimary,
//         title: const Text('Connection'),
//       ),
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: <Widget>[
//             Padding(
//               padding: const EdgeInsets.all(16.0),
//               child: TextFormField(
//                 controller: _iPController,
//                 decoration: const InputDecoration(
//                   labelText: 'Server IP Address',
//                 ),
//               ),
//             ),
//             Padding(
//               padding: const EdgeInsets.all(16.0),
//               child: TextFormField(
//                 controller: _nameController,
//                 decoration: const InputDecoration(
//                   labelText: 'Name',
//                 ),
//               ),
//             ),
//             ElevatedButton(
//               onPressed: () {
//                 String serverIP = _iPController.text;
//                 int serverPort = 5000; // Change this to the desired port number
//                 // Establish socket connection
//                 Socket.connect(serverIP, serverPort).then((Socket socket) {
//                   print('Connected to server');

//                   // Sending data to the server
//                   socket.write('Hello from Flutter!');

//                   // Listening for data from the server
//                   socket.listen((List<int> event) {
//                     print('Received: ${utf8.decode(event)}');
//                   }, onError: (error) {
//                     print('Error: $error');
//                     socket.destroy();
//                   }, onDone: () {
//                     print('Disconnected from server');
//                     socket.destroy();
//                   });
//                 }).catchError((error) {
//                   print('Error: $error');
//                 });
//               },
//               child: const Text('Submit'),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
