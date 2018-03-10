---
layout: post
comments: true
title:  "Multi-process proxy"
excerpt: "in progress"
date:   2017-01-20 11:00:00
mathjax: true
---

Assuming this code run in the middle server when client trys to connect to google.

## proxy.cpp

### void HTTPProxy::CreateServerSocket(int port)

1. create file stream in this process stack
2. configure port number, start accepting request and backlog

\\(\bullet \\) socket(): configure the right socket to get socket descriptor/file stream/file system struct pointer for incoming request connection

\\(\bullet \\) struct sockaddr_in serverAddr: form to fill in about server socket information about your address, namely, port and IP address

\\(\bullet \\) bind(): fill file system struct pointer with port number to associate socket/file stream with a unique port. port number is used by the kernel to match an incoming packet to a certain process's socket descriptor.

\\(\bullet \\) listen(): fill file system struct pointer with start accepting request and backlog. Backlog is the number of connections allowed on the incoming queue. Kernel makes incoming connection wait until you accept() them.

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



---







### void ProxyRequest()

Quarterback method called in main.cpp "request listening while loop" to spawn proxy request in multi-process fashion.

1. kernel creates new file stream for new incoming connection
2. receive and copy client request from stream to request_message
3. HTTP Request Parsing Library check request format
4. connect to google's file stream 
5. send request to google
6. proxy request back client


\\(\bullet \\)accept(): Once set SOCK_STREAM socket with bind() listen(), accept() reply to kernal to get the NEW client socket descriptor/file system pointer of new incoming connection. Accept() also store client's address and port in struct sockaddr_in clientAddr.

\\(\bullet \\)inet_ntoa(): convert network IP addresses from a dots-and-number format (e.g. "192.168.5.10") to a struct in_addr and back

\\(\bullet \\)ntohs(): network byte order to host byte order short, depending on sent data type

Different computers use different byte orderings internally for their multibyte integers. Intel did it the weird way "little-endian", and Motorola, IBM etc. "big-endian" byte orderings become prefered network byte ordering. If receving from PowerPC, ntohs() does nothing since already in Network Byte Order but if receiving from Intel machine ntohs() swap all the bytes around.

\\(\bullet \\)recv(): receive and copy client request from NEW client stream to request_message

\\(\bullet \\)CreateRemoteSocket(): return file stream of google server

\\(\bullet \\)SendRequestRemote(): send client's request to google

\\(\bullet \\)ProxyBackClient(): receive google's response then proxy back to client

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

### CreateRemoteSocket():

1. DNS lookups and write to struct addrinfo
2. configure google socket 
3. connect to google to get google file stream

\\(\bullet \\)getaddrinfo(): given address and port, DNS lookups and write to struct addrinfo

\\(\bullet \\)socket(): configure socket for google server

\\(\bullet \\)connect(): file system pointer connected to google's server

```c

int HTTPProxy::CreateRemoteSocket(char* remote_addr, char* port){
    // given address and port, configure hints to get results about host name
    struct addrinfo hints, *servinfo;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // use AF_INET6 to force IPv6
    hints.ai_socktype = SOCK_STREAM;
    
    if (getaddrinfo(remote_addr, port, &hints, &servinfo) !=0){
        cout << " Error in server address format ! \n" << endl;
    }

    // once get hostname info, creates remote socket n make a connection on socket
    int remote_socket;
    if((remote_socket = socket(servinfo->ai_family, servinfo->ai_socktype, servinfo->ai_protocol))<0) {
        cout << " Error in creating socket to server ! \n" << endl;
    }
    
    if(connect(remote_socket, servinfo->ai_addr, servinfo->ai_addrlen) <0){
        cout << " Error in connecting to server ! \n" << endl;
    }
    
    freeaddrinfo(servinfo);
    return remote_socket;
}
```


### SendRequestRemote(): 

\\(\bullet \\)send(): send client's request to google

```
void HTTPProxy::SendRequestRemote(const char *req_string, int remote_socket, int buff_length){
	string temp;
	temp.append(req_string);
    int totalsent = 0;
    int senteach;
    cout << "SendRequestRemote : "<< totalsent << " , " << buff_length << endl;
	while (totalsent < buff_length) {
        cout << "about to send to remote" << endl;
		if ((senteach = send(remote_socket, (void *) (req_string + totalsent), buff_length - totalsent, 0)) < 0) {
            cout << "error sending ot remote" << endl;
        }
        
        // senteach = send(remote_socket, (void *) (req_string + totalsent), buff_length - totalsent, 0);
        cout << "sent to remote" << senteach <<  endl;
		totalsent += senteach;
        cout << "total sent to remote: " << totalsent << endl;
	}	
}
```


### ProxyBackClient():

\\(\bullet \\)recv(): receive google's response


\\(\bullet \\)send(): send google's response to client


```
void HTTPProxy::ProxyBackClient(int client_fd, int remote_socket){
    int MAX_BUF_SIZE = 5000;
	int buff_length;
	char received_buf[MAX_BUF_SIZE];

    // receive from remote's response, send back to client
	while ((buff_length = recv(remote_socket, received_buf, MAX_BUF_SIZE, 0)) > 0) {
        cout << "received from remote: "<< buff_length << endl;
        int totalsent = 0;
        int senteach;
        while (totalsent < buff_length) {		
            if ((senteach = send(client_fd, (void *) (received_buf + totalsent), buff_length - totalsent, 0)) < 0) {                
                fprintf (stderr," Error in sending to server ! \n");
                    exit (1);
            }
            totalsent += senteach;
            cout << "sending back to client" << totalsent << endl;
		memset(received_buf,0,sizeof(received_buf));	
    	}      
    }
}
```


## blacklist.cpp


\\(\bullet \\)HTTPBlacklist::HTTPBlacklist(): constructor that reads blacklist.txt into a vector of regex

```
HTTPBlacklist::HTTPBlacklist(const char* file_name){
    ifstream infile(file_name);
    while(true){
        string line;
        getline(infile, line);
        regex re(line);
        blacklist_websites.push_back(re);
    }    
}
```

\\(\bullet \\)is_blacklisted(const char* website): check if the website in contained in vector of regex

```
bool HTTPBlacklist::is_blacklisted(const char* website){
    for (const regex& re: blacklist_websites){
        if (regex_match(website,re)){
            return true;
        }
    }
    return false;
}
```


## getpid(), fork(), wait(), execv()

fork(): duplicate current process and its memory space, if pid == 0 it is child, if pid value its parent
exec(): child process run a different process









## reference:
[Awesome C/C++ networking guide](https://beej.us/guide/bgnet/html/multi/index.html)