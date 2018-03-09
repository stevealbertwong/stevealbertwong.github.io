---
layout: post
comments: true
title:  "Multi-process proxy"
excerpt: "in progress"
date:   2017-01-14 11:00:00
mathjax: true
---


## proxy.cpp

### void HTTPProxy::CreateServerSocket(int port)

\\(\textbullet \\) socket(): configure the right socket to get mSocketDescriptor file stream

\\(\textbullet \\) setsockopt()

\\(\textbullet \\) serverAddr:

\\(\textbullet \\) bind():

\\(\textbullet \\) listen(): 

```
void HTTPProxy::CreateServerSocket(int port){

    mSocketDescriptor = socket(AF_INET, SOCK_STREAM, 0);
    
    const int optval = 1;
    // allow other sockets to bind to this port, "Address already in use" error 
    setsockopt(mSocketDescriptor, SOL_SOCKET, SO_REUSEADDR, &optval , sizeof(int)); 
    
    struct sockaddr_in serverAddr;
    bzero(&serverAddr, sizeof(serverAddr)); // memset
    serverAddr.sin_family = AF_INET;
    // automatically be filled with current host's IP address
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY); 
    serverAddr.sin_port = htons(port);
    struct sockaddr *sa = (struct sockaddr *) &serverAddr;
    
    ::bind(mSocketDescriptor, sa, sizeof(serverAddr)); // c libs instead of std::bind
    
    const size_t kMaxQueuedRequests = 128;
    listen(mSocketDescriptor, kMaxQueuedRequests);
    cout << "listening on port: " << port << endl;    
}
```











### void ProxyRequest()

Method called in main.cpp request listening while loop to spawn proxy request in multi-process fashion.

1. receive client's request
2. forward client's request to google
3. receive and copy client request from stream to buf

\\(\textbullet \\)accept(): Once you've gone through the trouble of getting a SOCK_STREAM socket and setting it up for incoming connections with listen(), then you call accept() to actually get yourself a new socket descriptor to use for subsequent communication with the newly connected client.

\\(\textbullet \\)inet_ntoa(): convert network IP addresses from a dots-and-number format (e.g. "192.168.5.10") to a struct in_addr and back

\\(\textbullet \\)ntohs(): network byte order to host byte order short, depending on sent data type

Different computers use different byte orderings internally for their multibyte integers. Intel did it the weird way "little-endian", and Motorola, IBM etc. "big-endian" byte orderings become prefered network byte ordering. If receving from PowerPC, ntohs() does nothing since already in Network Byte Order but if receiving from Intel machine ntohs() swap all the bytes around.

\\(\textbullet \\)strstr(request_message, "\r\n\r\n"): return a ptr of first occurence of "\r\n\r\n" in request_message, loop thru until request_message until \r\n\r\n

\\(\textbullet \\)recv(): Once socket up and connected, read incoming data on TCP SOCK_STREAM sockets

```
void HTTPProxy::ProxyRequest(){
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    // write incoming client's connection to sockaddr
    int client_fd = accept(mSocketDescriptor, (struct sockaddr *) &clientAddr, &clientAddrSize);
    
    const char *clientIPAddress = inet_ntoa(clientAddr.sin_addr);
    uint16_t clientPort = ntohs(clientAddr.sin_port);    
    cout << "server got connection from client:" << clientIPAddress << clientPort << endl;



    // forward client request to google
    int MAX_BUFFER_SIZE = 5000;
    char buf[MAX_BUFFER_SIZE];
    char *request_message = (char *) malloc(MAX_BUFFER_SIZE); 
    request_message[0] = '\0';
	int total_received_bits = 0;
    
    // receive n copy client request from stream to buf
    while (strstr(request_message, "\r\n\r\n") == NULL) {
        int byte_recvd = recv(client_fd, buf, MAX_BUFFER_SIZE, 0);
        total_received_bits += byte_recvd;
        buf[byte_recvd] = '\0';
	  	if (total_received_bits > MAX_BUFFER_SIZE) {
			MAX_BUFFER_SIZE *= 2;
			request_message = (char *) realloc(request_message, MAX_BUFFER_SIZE);
        }
        strcat(request_message, buf);
    }
    cout << "request_message : " << request_message << endl;
    


    struct ParsedRequest *req;    // contains parsed request
    req = ParsedRequest_create();    
    
    if(ParsedRequest_parse(req, request_message, strlen(request_message))<0){
        cout << "request message format not supported yet" << endl;
        
    } else {
        if (req->port == NULL) {
            req->port = (char *) "80";
        }		 
        char* req_string = RequestToString(req);	
        cout << "req_string : " << req_string << endl;


        cout << "client host n port: " << req->host << req->port << endl;
        // remote socket: connection to remote host e.g. google
        int remote_socket = CreateRemoteSocket(req->host, req->port);
        
        cout << "SendRequestRemote: " << remote_socket << " total received bits" << total_received_bits << endl;
        SendRequestRemote(req_string, remote_socket, total_received_bits);

        cout << "ProxyBackClient" << endl;
        ProxyBackClient(client_fd, remote_socket);
        
        ParsedRequest_destroy(req);		
        close(client_fd);   
        close(remote_socket);
    }
}
```


## reference:
[Awesome C/C++ networking guide](https://beej.us/guide/bgnet/html/multi/index.html)