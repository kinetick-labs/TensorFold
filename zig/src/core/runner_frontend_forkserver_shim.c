typedef unsigned long size_t;
typedef long ssize_t;
typedef int pid_t;
struct rlimit { unsigned long cur, max; };
extern char **environ;
extern int fork(void);
extern pid_t waitpid(pid_t, int *, int);
extern pid_t getpid(void);
extern ssize_t read(int, void *, size_t);
extern ssize_t write(int, const void *, size_t);
extern int chdir(const char *);
extern int open(const char *, int, ...);
extern int dup2(int, int);
extern int close(int);
extern void _exit(int) __attribute__((noreturn));
extern void exit(int) __attribute__((noreturn));
extern int on_exit(void (*)(int, void *), void *);
extern int fflush(void *);
extern char *getenv(const char *);
extern int unsetenv(const char *);
extern int setrlimit(int, const struct rlimit *);
extern void *mmap(void *, size_t, int, int, int, long);
extern void *dlsym(void *, const char *);
extern int *__errno_location(void);
extern long syscall(long, ...);
typedef int (*main_fn)(int, char **, char **);
typedef int (*start_fn)(main_fn, int, char **, void (*)(void), void (*)(void), void (*)(void), void *);
static main_fn tool_main;
static int control = -1;
static volatile int *flags;
static char request[65536];
static char *args[1024];
static int full(int fd, void *buffer, size_t count, int writing) {
    char *p = buffer;
    while (count) {
        ssize_t n = writing ? write(fd, p, count) : read(fd, p, count);
        if (n < 0 && *__errno_location() == 4) continue;
        if (n <= 0) return 0;
        p += n;
        count -= (size_t)n;
    }
    return 1;
}
static void report(int rc, void *unused) {
    (void)unused;
    int status = (rc & 0xff) << 8;
    fflush(0);
    close(1);
    close(2);
    if (__sync_bool_compare_and_swap(&flags[2], 0, 1)) full(control, &status, 4, 1);
}
static void child(void) {
    unsigned int size;
    if (!full(control, &size, 4, 0) || size < 6 || size > sizeof request - 1) _exit(0);
    flags[0] = 1;
    if (!full(control, request, size, 0)) _exit(0);
    request[size] = 0;
    unsigned int limit = *(unsigned int *)request;
    char *dir = request + 4, *p = dir;
    while (*p) ++p;
    ++p;
    int n = 0;
    while (p < request + size && n < 1023) {
        args[n++] = p;
        while (*p) ++p;
        ++p;
    }
    args[n] = 0;
    int pid = getpid();
    if (!full(control, &pid, 4, 1)) _exit(0);
    flags[1] = 1;
    unsetenv("TF_FORKSERVER_FD");
    unsetenv("LD_PRELOAD");
    if (n == 0 || chdir(dir) != 0) _exit(127);
    int in = open("/dev/null", 0), out = open(".stdout", 01101, 0600), err = open(".stderr", 01101, 0600);
    if (in < 0 || out < 0 || err < 0 || dup2(in, 0) < 0 || dup2(out, 1) < 0 || dup2(err, 2) < 0) _exit(127);
    struct rlimit fsize = { limit, limit };
    setrlimit(1, &fsize);
    on_exit(report, 0);
    exit(tool_main(n, args, environ));
}
static int serve(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    // the server outlives calls: drop the runner's other pipes and sockets (close_range, else one by one)
    if (syscall(436, 4u, ~0u, 0u) != 0) for (int fd = 4; fd < 1024; ++fd) close(fd);
    flags = mmap(0, 4096, 3, 0x21, -1, 0);
    if (flags == (void *)-1l) _exit(0);
    unsigned int ready = 0x52454459;
    if (!full(control, &ready, 4, 1)) _exit(0);
    for (;;) {
        flags[0] = flags[1] = flags[2] = 0;
        int pid = fork();
        if (pid == 0) child();
        if (pid < 0) _exit(0);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && *__errno_location() == 4) {}
        if (!flags[0]) _exit(0);
        int zero = 0;
        if (!flags[1] && !full(control, &zero, 4, 1)) _exit(0);
        if (__sync_bool_compare_and_swap(&flags[2], 0, 1) && !full(control, &status, 4, 1)) _exit(0);
    }
}
int __libc_start_main(main_fn main, int argc, char **argv, void (*init)(void), void (*fini)(void),
                      void (*rtld)(void), void *stack) {
    start_fn next = (start_fn)dlsym((void *)-1l, "__libc_start_main");
    const char *fd = getenv("TF_FORKSERVER_FD");
    tool_main = main;
    if (fd && *fd >= '3' && *fd <= '9' && !fd[1]) {
        control = *fd - '0';
        return next(serve, argc, argv, init, fini, rtld, stack);
    }
    return next(main, argc, argv, init, fini, rtld, stack);
}
