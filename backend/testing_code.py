import requests

url = 'http://your_ip_address:port_number/data'
with open("your/filepath/here", 'rb') as file:
    files = {'messageFile': file}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.text)