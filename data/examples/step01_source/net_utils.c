// ============================================================
// STEP 1: Source Code → Compiled Binary with Debug Symbols
// ============================================================
// INPUT:  C source code (e.g., from coreutils or custom project)
// OUTPUT: Binary with debug symbols (compiled with gcc -g -O2)
// ============================================================

// Example source file: src/net_utils.c

#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>

void socket_init(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(fd, 5);
}

int parse_config(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char buf[256];
    int count = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        count++;
    }
    fclose(fp);
    return count;
}

void handle_client(int client_fd) {
    char buffer[1024];
    int n = read(client_fd, buffer, sizeof(buffer));
    if (n > 0) {
        write(client_fd, buffer, n);  // echo back
    }
    close(client_fd);
}

size_t compute_checksum(const unsigned char *data, size_t len) {
    size_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += data[i];
        sum = (sum << 3) | (sum >> (sizeof(size_t)*8 - 3));
        sum ^= 0xA5A5A5A5;
    }
    return sum;
}

// ============================================================
// COMPILATION COMMANDS:
//
//   # Compile with debug symbols
//   gcc -g -O2 -o net_utils_sym net_utils.c
//
//   # Verify symbols are present
//   nm -n net_utils_sym | head -20
//
// EXPECTED nm OUTPUT:
//   0000000000401230 T socket_init
//   00000000004012a0 T parse_config
//   0000000000401350 T handle_client
//   0000000000401400 T compute_checksum
//
// ============================================================
