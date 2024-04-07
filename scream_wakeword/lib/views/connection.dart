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
