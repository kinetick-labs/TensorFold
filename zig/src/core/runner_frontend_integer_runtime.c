// RV32 integer library; program_compile supplies memory operations, putchar and startup.
typedef unsigned int tf_wu;
typedef unsigned long long tf_wd;
struct tf_wint { tf_wu v[8]; int neg; };

// ABI: nodes are topological with n <= 64; tf_run prints node n-1 and tf_run_many takes ordered output refs.
struct tf_node { int op, l, r, k; const char *text; };

// Ops 0..15 are stable; 16..21 compare integers and internal op 22 reads a cached division remainder.
static volatile struct tf_wint tf_wvalues[64];
static volatile struct tf_wint tf_wremainders[64];

// The parent proves every admitted input and node result has magnitude below 10^62; scratch products use 512 bits.
static int tf_wbad;

static struct tf_wint tf_wsmall(tf_wu x) {
    struct tf_wint a;
    for (int i = 0; i < 8; ++i) a.v[i] = 0;
    a.v[0] = x;
    a.neg = 0;
    return a;
}

static int tf_wused(const struct tf_wint *a) {
    int n = 8;
    while (n && !a->v[n - 1]) --n;
    return n;
}

static void tf_wnormalize(struct tf_wint *a) {
    if (!tf_wused(a)) a->neg = 0;
}

static int tf_wcmp(const struct tf_wint *a, const struct tf_wint *b) {
    for (int i = 7; i >= 0; --i) {
        if (a->v[i] != b->v[i]) return a->v[i] > b->v[i] ? 1 : -1;
    }
    return 0;
}

static void tf_wbound(const struct tf_wint *a) {
    const struct tf_wint limit = {
        { 0u, 1073741824u, 4155282634u, 576837993u, 895677624u, 3947553080u, 15930u, 0u }, 0
    };
    if (tf_wcmp(a, &limit) >= 0) tf_wbad = 1;
}

static struct tf_wint tf_waddmag(const struct tf_wint *a, const struct tf_wint *b) {
    struct tf_wint c = tf_wsmall(0);
    tf_wd carry = 0;
    for (int i = 0; i < 8; ++i) {
        tf_wd s = (tf_wd)a->v[i] + b->v[i] + carry;
        c.v[i] = (tf_wu)s;
        carry = s >> 32;
    }
    if (carry) tf_wbad = 1;
    return c;
}

// Magnitude subtraction also supports the wrapped low words of a 257-bit division remainder.
static struct tf_wint tf_wsubmag(const struct tf_wint *a, const struct tf_wint *b) {
    struct tf_wint c = tf_wsmall(0);
    tf_wd borrow = 0;
    for (int i = 0; i < 8; ++i) {
        tf_wd t = (tf_wd)b->v[i] + borrow;
        c.v[i] = a->v[i] - (tf_wu)t;
        borrow = (tf_wd)a->v[i] < t;
    }
    return c;
}

static struct tf_wint tf_wadd(struct tf_wint a, struct tf_wint b) {
    struct tf_wint c;
    if (a.neg == b.neg) {
        c = tf_waddmag(&a, &b);
        c.neg = a.neg;
    } else if (tf_wcmp(&a, &b) >= 0) {
        c = tf_wsubmag(&a, &b);
        c.neg = a.neg;
    } else {
        c = tf_wsubmag(&b, &a);
        c.neg = b.neg;
    }
    tf_wnormalize(&c);
    return c;
}

static void tf_wproduct(tf_wu w[16], const struct tf_wint *a, const struct tf_wint *b) {
    for (int i = 0; i < 16; ++i) w[i] = 0;
    int na = tf_wused(a), nb = tf_wused(b);
    for (int i = 0; i < na; ++i) {
        tf_wd carry = 0;
        for (int j = 0; j < nb; ++j) {
            tf_wd t = (tf_wd)a->v[i] * b->v[j] + w[i + j] + carry;
            w[i + j] = (tf_wu)t;
            carry = t >> 32;
        }
        w[i + nb] = (tf_wu)carry;
    }
}

static struct tf_wint tf_wmul(const struct tf_wint *a, const struct tf_wint *b) {
    tf_wu w[16];
    tf_wproduct(w, a, b);
    struct tf_wint c = tf_wsmall(0);
    for (int i = 0; i < 8; ++i) {
        c.v[i] = w[i];
        if (w[i + 8]) tf_wbad = 1;
    }
    c.neg = a->neg != b->neg;
    tf_wnormalize(&c);
    tf_wbound(&c);
    return c;
}

// All divisions use 32-bit instructions; the small-divisor path handles two halfwords per limb.
static tf_wu tf_wdivsmall(tf_wu *w, int n, tf_wu d) {
    tf_wu rem = 0;
    for (int i = n - 1; i >= 0; --i) {
        tf_wu x = w[i];
        if (d <= 65535u) {
            tf_wu t = (rem << 16) | (x >> 16);
            tf_wu q = t / d;
            t = ((t % d) << 16) | (x & 65535u);
            w[i] = (q << 16) | (t / d);
            rem = t % d;
        } else if (!rem) {
            w[i] = x / d;
            rem = x % d;
        } else {
            tf_wu q = 0;
            for (int bit = 31; bit >= 0; --bit) {
                tf_wu high = rem >> 31;
                rem = (rem << 1) | ((x >> bit) & 1u);
                if (high || rem >= d) {
                    rem -= d;
                    q |= 1u << bit;
                }
            }
            w[i] = q;
        }
    }
    return rem;
}

static int tf_wbits(const tf_wu *w, int n) {
    while (n && !w[n - 1]) --n;
    if (!n) return 0;
    tf_wu x = w[n - 1];
    int bits = (n - 1) * 32;
    while (x) { ++bits; x >>= 1; }
    return bits;
}

static int tf_wstep(struct tf_wint *r, tf_wu bit, const struct tf_wint *d) {
    tf_wu carry = bit;
    for (int i = 0; i < 8; ++i) {
        tf_wu high = r->v[i] >> 31;
        r->v[i] = (r->v[i] << 1) | carry;
        carry = high;
    }
    if (!carry && tf_wcmp(r, d) < 0) return 0;
    *r = tf_wsubmag(r, d);
    return 1;
}

static void tf_wdivmag(const struct tf_wint *a, const struct tf_wint *b,
                       struct tf_wint *q, struct tf_wint *r) {
    *q = tf_wsmall(0);
    *r = tf_wsmall(0);
    int na = (tf_wbits(a->v, 8) + 15) / 16;
    int nb = (tf_wbits(b->v, 8) + 15) / 16;
    if (!nb) { tf_wbad = 1; return; }
    if (tf_wcmp(a, b) < 0) { *r = *a; r->neg = 0; return; }
    if (nb == 1) {
        *q = *a;
        q->neg = 0;
        r->v[0] = tf_wdivsmall(q->v, tf_wused(q), b->v[0]);
        return;
    }
    tf_wu u[17], v[16];
    int shift = 16 * nb - tf_wbits(b->v, 8);
    tf_wu carry = 0;
    for (int i = 0; i < nb; ++i) {
        tf_wu x = ((b->v[i / 2] >> (16 * (i % 2))) & 65535u) << shift;
        v[i] = (x | carry) & 65535u;
        carry = x >> 16;
    }
    carry = 0;
    for (int i = 0; i < na; ++i) {
        tf_wu x = ((a->v[i / 2] >> (16 * (i % 2))) & 65535u) << shift;
        u[i] = (x | carry) & 65535u;
        carry = x >> 16;
    }
    u[na] = carry;
    for (int j = na - nb; j >= 0; --j) {
        tf_wu top = (u[j + nb] << 16) | u[j + nb - 1];
        tf_wu digit = u[j + nb] == v[nb - 1] ? 65535u : top / v[nb - 1];
        tf_wu rem = top - digit * v[nb - 1];
        while (rem < 65536u && digit * v[nb - 2] > (rem << 16) + u[j + nb - 2]) {
            --digit;
            rem += v[nb - 1];
        }
        tf_wu borrow = 0;
        for (int i = 0; i < nb; ++i) {
            tf_wu product = digit * v[i] + borrow;
            tf_wu low = product & 65535u;
            borrow = (product >> 16) + (u[j + i] < low);
            u[j + i] = (u[j + i] - low) & 65535u;
        }
        int negative = u[j + nb] < borrow;
        u[j + nb] = (u[j + nb] - borrow) & 65535u;
        if (negative) {
            --digit;
            carry = 0;
            for (int i = 0; i < nb; ++i) {
                tf_wu sum = u[j + i] + v[i] + carry;
                u[j + i] = sum & 65535u;
                carry = sum >> 16;
            }
            u[j + nb] = (u[j + nb] + carry) & 65535u;
        }
        q->v[j / 2] |= digit << (16 * (j % 2));
    }
    for (int i = 0; i < nb; ++i) {
        tf_wu digit = ((u[i] | (u[i + 1] << 16)) >> shift) & 65535u;
        r->v[i / 2] |= digit << (16 * (i % 2));
    }
}

static void tf_wdivmod(const struct tf_wint *a, const struct tf_wint *b,
                       struct tf_wint *q, struct tf_wint *r) {
    tf_wdivmag(a, b, q, r);
    if (tf_wbad) return;
    if (a->neg != b->neg && tf_wused(r)) {
        struct tf_wint one = tf_wsmall(1);
        *q = tf_waddmag(q, &one);
        *r = tf_wsubmag(b, r);
    }
    q->neg = a->neg != b->neg;
    r->neg = b->neg;
    tf_wnormalize(q);
    tf_wnormalize(r);
}

static struct tf_wint tf_wmodmul(const struct tf_wint *a, const struct tf_wint *b,
                                const struct tf_wint *m) {
    tf_wu w[16];
    tf_wproduct(w, a, b);
    struct tf_wint r = tf_wsmall(0);
    if (tf_wused(m) == 1) {
        r.v[0] = tf_wdivsmall(w, 16, m->v[0]);
        return r;
    }
    for (int bit = tf_wbits(w, 16) - 1; bit >= 0; --bit)
        tf_wstep(&r, (w[bit / 32] >> (bit % 32)) & 1u, m);
    return r;
}

static struct tf_wint tf_wpow(struct tf_wint a, const struct tf_wint *e,
                             const struct tf_wint *mod) {
    struct tf_wint result = tf_wsmall(1), m = tf_wsmall(0), unused;
    if (e->neg) { tf_wbad = 1; return m; }
    if (mod) {
        m = *mod;
        m.neg = 0;
        if (!tf_wused(&m)) { tf_wbad = 1; return m; }
        if (tf_wused(&m) == 1 && m.v[0] == 1) return tf_wsmall(0);
        int negative = a.neg;
        struct tf_wint reduced;
        tf_wdivmag(&a, &m, &unused, &reduced);
        a = reduced;
        if (negative && tf_wused(&a)) a = tf_wsubmag(&m, &a);
    }
    int bits = tf_wbits(e->v, 8);
    for (int bit = 0; bit < bits && !tf_wbad; ++bit) {
        if ((e->v[bit / 32] >> (bit % 32)) & 1u)
            result = mod ? tf_wmodmul(&result, &a, &m) : tf_wmul(&result, &a);
        if (bit + 1 < bits) a = mod ? tf_wmodmul(&a, &a, &m) : tf_wmul(&a, &a);
    }
    if (mod && mod->neg && tf_wused(&result)) {
        result = tf_wsubmag(&m, &result);
        result.neg = 1;
    }
    return result;
}

static struct tf_wint tf_wgcd(struct tf_wint a, struct tf_wint b) {
    a.neg = b.neg = 0;
    while (tf_wused(&b) && !tf_wbad) {
        struct tf_wint q, r;
        tf_wdivmag(&a, &b, &q, &r);
        a = b;
        b = r;
    }
    return a;
}

static struct tf_wint tf_wisqrt(const struct tf_wint *a) {
    struct tf_wint x = tf_wsmall(0), q, r;
    if (a->neg) { tf_wbad = 1; return x; }
    int bits = tf_wbits(a->v, 8);
    if (!bits) return x;
    int shift = (bits + 1) / 2;
    x.v[shift / 32] = 1u << (shift % 32);
    tf_wdivmag(a, &x, &q, &r);
    struct tf_wint y = tf_waddmag(&x, &q);
    tf_wdivsmall(y.v, 8, 2);
    while (tf_wcmp(&y, &x) < 0 && !tf_wbad) {
        x = y;
        tf_wdivmag(a, &x, &q, &r);
        y = tf_waddmag(&x, &q);
        tf_wdivsmall(y.v, 8, 2);
    }
    return x;
}

static struct tf_wint tf_wfactorial(const struct tf_wint *a) {
    struct tf_wint result = tf_wsmall(1), maximum = tf_wsmall(48);
    if (a->neg || tf_wcmp(a, &maximum) > 0) { tf_wbad = 1; return result; }
    for (tf_wu i = 2; i <= a->v[0] && !tf_wbad; ++i) {
        struct tf_wint factor = tf_wsmall(i);
        result = tf_wmul(&result, &factor);
    }
    return result;
}

static int tf_wsignedcmp(const struct tf_wint *a, const struct tf_wint *b) {
    if (a->neg != b->neg) return a->neg ? -1 : 1;
    return a->neg ? -tf_wcmp(a, b) : tf_wcmp(a, b);
}

static struct tf_wint tf_wrange(struct tf_wint a, const struct tf_wint *stop,
                               const struct tf_wint *step) {
    struct tf_wint result = tf_wsmall(1);
    if (!tf_wused(step)) { tf_wbad = 1; return result; }
    if (step->neg ? tf_wsignedcmp(&a, stop) <= 0 : tf_wsignedcmp(&a, stop) >= 0) return result;
    struct tf_wint negative = step->neg ? *stop : a;
    negative.neg = !negative.neg;
    struct tf_wint distance = tf_wadd(step->neg ? a : *stop, negative);
    struct tf_wint stride = *step, length, remainder;
    stride.neg = 0;
    tf_wdivmod(&distance, &stride, &length, &remainder);
    if (tf_wused(&remainder)) length = tf_wadd(length, tf_wsmall(1));
    struct tf_wint maximum = tf_wsmall(10000);
    if (tf_wbad || tf_wcmp(&length, &maximum) > 0) { tf_wbad = 1; return result; }
    int count = 0;
    while (!tf_wbad && (step->neg ? tf_wsignedcmp(&a, stop) > 0 : tf_wsignedcmp(&a, stop) < 0)) {
        if (++count > 10000) { tf_wbad = 1; break; }
        tf_wbound(&a);
        result = tf_wmul(&result, &a);
        if (!tf_wused(&result)) return result;
        a = tf_wadd(a, *step);
    }
    return result;
}

static struct tf_wint tf_wdigitsum(struct tf_wint a) {
    tf_wu sum = 0;
    if (a.neg) tf_wbad = 1;
    while (tf_wused(&a)) sum += tf_wdivsmall(a.v, 8, 10);
    return tf_wsmall(sum);
}

// Literal text contains signed digits without a radix prefix; k is radix 2, 10, or 16, not a child index.
static struct tf_wint tf_wparse(const volatile char *s, int base) {
    struct tf_wint a = tf_wsmall(0);
    if (!s || (base != 2 && base != 10 && base != 16)) { tf_wbad = 1; return a; }
    int negative = *s == '-';
    if (*s == '-' || *s == '+') ++s;
    if (!*s) tf_wbad = 1;
    int used = 1;
    while (*s) {
        tf_wu chunk = 0, multiplier = 1;
        int count = 0;
        do {
            unsigned char c = (unsigned char)*s++;
            int digit;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else { tf_wbad = 1; return a; }
            if (digit >= base) { tf_wbad = 1; return a; }
            chunk = chunk * (tf_wu)base + (tf_wu)digit;
            multiplier *= (tf_wu)base;
            ++count;
        } while (base == 10 && count < 4 && *s);
        tf_wd carry = chunk;
        for (int i = 0; i < used; ++i) {
            tf_wd t = (tf_wd)a.v[i] * multiplier + carry;
            a.v[i] = (tf_wu)t;
            carry = t >> 32;
        }
        if (carry) {
            if (used == 8) { tf_wbad = 1; return a; }
            a.v[used++] = (tf_wu)carry;
        }
    }
    a.neg = negative;
    tf_wnormalize(&a);
    return a;
}

static int tf_wformat(struct tf_wint a, char text[79]) {
    char digits[78];
    int count = 0, len = 0;
    int used = tf_wused(&a);
    do {
        tf_wu chunk = tf_wdivsmall(a.v, used, 10000);
        while (used && !a.v[used - 1]) --used;
        int width = 0;
        do {
            digits[count++] = (char)('0' + chunk % 10);
            chunk /= 10;
            ++width;
        } while (used ? width < 4 : chunk != 0);
    } while (used);
    if (a.neg) text[len++] = '-';
    while (count) text[len++] = digits[--count];
    return len;
}

static void tf_wprint(const struct tf_wint *a) {
    char text[79];
    int len = tf_wformat(*a, text);
    for (int i = 0; i < len; ++i) putchar(text[i]);
}

// Fixed Decimal formatting inserts the proved scale into a runtime coefficient.
static void tf_wfixed(const struct tf_wint *value, int scale) {
    char text[79];
    int length = tf_wformat(*value, text);
    int start = text[0] == '-', digits = length - start;
    if (start) putchar('-');
    if (!scale) {
        for (int i = start; i < length; ++i) putchar(text[i]);
    } else if (digits <= scale) {
        putchar('0'); putchar('.');
        for (int i = digits; i < scale; ++i) putchar('0');
        for (int i = start; i < length; ++i) putchar(text[i]);
    } else {
        for (int i = start; i < length; ++i) {
            if (i == length - scale) putchar('.');
            putchar(text[i]);
        }
    }
}

// Evaluate every node before printing any ordered root; volatile reads retain the requested runtime arithmetic.
int tf_run_formatted(const volatile struct tf_node *nodes, int n, const volatile int *roots,
                     const volatile unsigned char *endings, const volatile unsigned char *styles,
                     const volatile unsigned char *scales, int count) {
    tf_wbad = 0;
    if (!nodes || n < 1 || n > 64 || !roots || count < 1 || count > 8) return 1;
    int outputs[8];
    unsigned char separators[8], formats[8], precision[8];
    for (int j = 0; j < count; ++j) {
        outputs[j] = roots[j];
        if (outputs[j] < 0 || outputs[j] >= n) return 1;
        separators[j] = endings ? endings[j] : '\n';
        if (separators[j] != ' ' && separators[j] != '\n') return 1;
        formats[j] = styles ? styles[j] : 0;
        precision[j] = scales ? scales[j] : 0;
        if (formats[j] > 2 || precision[j] > 6) return 1;
        int op = nodes[outputs[j]].op;
        if (formats[j] && (op == 10 || (op >= 16 && op <= 21))) return 1;
    }
    for (int i = 0; i < n; ++i) {
        const volatile struct tf_node *node = &nodes[i];
        struct tf_wint a = tf_wsmall(0), b = tf_wsmall(0), c;
        int op = node->op, l = node->l, r = node->r, k = node->k;
        if (op < 0 || op > 22) return 1;
        if (op) {
            if (l < 0 || l >= i || (op != 22 && nodes[l].op == 10)) return 1;
            a = tf_wvalues[l];
        }
        if ((op >= 2 && op <= 8) || op == 10 || op == 11 || op == 14 || (op >= 16 && op <= 21)) {
            if (r < 0 || r >= i || nodes[r].op == 10) return 1;
            b = tf_wvalues[r];
        }
        switch (op) {
        case 0: c = tf_wparse(node->text, k); break;
        case 1:
            c = a;
            c.neg = !c.neg;
            tf_wnormalize(&c);
            break;
        case 2: c = tf_wadd(a, b); break;
        case 3:
            b.neg = !b.neg;
            c = tf_wadd(a, b);
            break;
        case 4: c = tf_wmul(&a, &b); break;
        case 5: case 6: case 10: {
            struct tf_wint remainder;
            tf_wdivmod(&a, &b, &c, &remainder);
            tf_wremainders[i] = remainder;
            if (op == 6) c = remainder;
            break;
        }
        case 7: c = tf_wpow(a, &b, (const struct tf_wint *)0); break;
        case 8: case 14: {
            if (k < 0 || k >= i || nodes[k].op == 10) return 1;
            struct tf_wint third = tf_wvalues[k];
            c = op == 8 ? tf_wpow(a, &b, &third) : tf_wrange(a, &b, &third);
            break;
        }
        case 9: {
            char text[79];
            c = tf_wsmall((tf_wu)tf_wformat(a, text));
            break;
        }
        case 11: c = tf_wgcd(a, b); break;
        case 12: c = tf_wisqrt(&a); break;
        case 13: c = tf_wfactorial(&a); break;
        case 15: c = tf_wdigitsum(a); break;
        case 16: case 17: case 18: case 19: case 20: case 21: {
            int cmp = tf_wsignedcmp(&a, &b);
            c = tf_wsmall((tf_wu)(op == 16 ? cmp == 0 : op == 17 ? cmp != 0 :
                op == 18 ? cmp < 0 : op == 19 ? cmp <= 0 : op == 20 ? cmp > 0 : cmp >= 0));
            break;
        }
        case 22:
            if (nodes[l].op != 5 && nodes[l].op != 10) return 1;
            c = tf_wremainders[l];
            break;
        default: return 1;
        }
        tf_wbound(&c);
        if (tf_wbad) return 1;
        tf_wvalues[i] = c;
    }
    for (int j = 0; j < count; ++j) {
        int root = outputs[j], op = nodes[root].op;
        struct tf_wint value = tf_wvalues[root];
        if (formats[j] >= 2) {
            tf_wfixed(&value, precision[j]);
        } else if (formats[j] == 1) {
            tf_wprint(&value); putchar('.'); putchar('0');
        } else if (op >= 16 && op <= 21) {
            const char *text = tf_wused(&value) ? "True" : "False";
            while (*text) putchar(*text++);
        } else {
            if (op == 10) putchar('(');
            tf_wprint(&value);
            if (op == 10) {
                putchar(',');
                putchar(' ');
                struct tf_wint remainder = tf_wremainders[root];
                tf_wprint(&remainder);
                putchar(')');
            }
        }
        putchar(separators[j]);
    }
    return 0;
}

int tf_run_prints(const volatile struct tf_node *nodes, int n, const volatile int *roots,
                  const volatile unsigned char *endings, int count) {
    return tf_run_formatted(nodes, n, roots, endings, (const volatile unsigned char *)0,
                            (const volatile unsigned char *)0, count);
}

int tf_run_many(const volatile struct tf_node *nodes, int n, const volatile int *roots, int count) {
    return tf_run_prints(nodes, n, roots, (const volatile unsigned char *)0, count);
}

// Retain the standalone ABI and its last-node output convention.
int tf_run(const volatile struct tf_node *nodes, int n) {
    if (n < 1 || n > 64) return 1;
    const int root = n - 1;
    return tf_run_many(nodes, n, &root, 1);
}
