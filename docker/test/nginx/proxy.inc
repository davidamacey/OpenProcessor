# Server-Sent Events need no buffering and a long read timeout; HTTP/1.1 to the upstream with an
# empty Connection header keeps the stream open (nginx docs, ngx_http_proxy_module).
location / {
    proxy_pass $upstream;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    client_max_body_size 64m;
}
