typedef struct { int tag, x; } V;
typedef struct { int start, len, cap; } Obj;
static V cells[1536];
static Obj objects[96];
static char chars[2048];
static int nc, no, ns;
__attribute__((always_inline)) static inline V py_hold(V v) {
 __asm__ volatile("" : "+r"(v.x)); return v;
}
__attribute__((noreturn)) static void fail(void) {
 register int a0 __asm__("a0")=1; register int a7 __asm__("a7")=93;
 __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory"); for(;;) {}
}
static V val(int tag,int x) { V v={tag,x}; return py_hold(v); }
static V num(int x) { return val(0,x); }
static V boolean(int x) { return val(1,!!x); }
static V none(void) { return val(2,0); }
static V checked(V v) { if(v.tag<0) fail(); return py_hold(v); }
static int scalar(V v) { v=checked(v); if(v.tag>1) fail(); return v.x; }
static Obj *obj(V v) { if(v.tag<3||v.tag>15||v.tag==9||v.x<0||v.x>=no) fail(); return objects+v.x; }
static int length(V v) {
 if(v.tag==9||v.tag>=13)fail(); return obj(v)->len;
}
static int truth(V v) { v=checked(v); return v.tag==2?0:v.tag<2?v.x!=0:v.tag>=13||v.tag==9?1:length(v)!=0; }
static V fresh(int tag) {
 if(no==96) fail(); int i=no++; objects[i]=(Obj){0,0,0}; return val(tag,i);
}
static void reserve(V v,int n) {
 Obj *o=obj(v); int limit=v.tag==6||v.tag==7?64:128;
 if(n>limit||n<0) fail(); if(n<=o->cap) return;
 int cap=o->cap?o->cap:4; while(cap<n) cap*=2;
 int width=v.tag==6?2:1; if(nc+cap*width>1536) fail();
 int start=nc; nc+=cap*width;
 for(int i=0;i<o->len*width;i++) cells[start+i]=cells[o->start+i];
 o->start=start; o->cap=cap;
}
static V text(const volatile char *s,int n) {
 if(n<0||n>128||ns+n>2048) fail(); V v=fresh(3); Obj *o=obj(v);
 o->start=ns; o->len=n; o->cap=n;
 for(int i=0;i<n;i++) chars[ns++]=(char)py_hold(num(s[i])).x;
 return v;
}
static V emptytext(void) { return text("",0); }
static void append(V v,V x) {
 if(v.tag!=4&&v.tag!=5&&v.tag!=7&&v.tag!=8) fail(); Obj *o=obj(v);
 reserve(v,o->len+1); cells[o->start+o->len++]=checked(x);
}
static int equal(V a,V b);
static int lookup(V v,V k) {
 if(k.tag!=0&&k.tag!=1&&k.tag!=3) fail(); Obj *o=obj(v); int w=v.tag==6?2:1;
 for(int i=0;i<o->len;i++) if(equal(cells[o->start+i*w],k)) return i;
 return -1;
}
static void store(V v,V k,V x) {
 if(v.tag!=6) fail(); Obj *o=obj(v); int i=lookup(v,k);
 if(i<0) { reserve(v,o->len+1); i=o->len++; cells[o->start+2*i]=k; }
 cells[o->start+2*i+1]=checked(x);
}
static void setadd(V v,V k) { if(v.tag!=7) fail(); if(lookup(v,k)<0) append(v,k); }
static int norm(int i,int n) { if(i<0)i+=n; if(i<0||i>=n)fail(); return i; }
static V at(V v,int i) {
 Obj *o=obj(v); i=norm(i,o->len);
 if(v.tag==3) return text((volatile char *)(chars+o->start+i),1);
 if(v.tag==7||v.tag>=13) fail();
 if(v.tag==10) {
  V pair=fresh(5);append(pair,py_hold(cells[o->start+2*i]));
  append(pair,py_hold(cells[o->start+2*i+1]));return pair;
 }
 if(v.tag==11)return py_hold(cells[o->start+2*i]);
 if(v.tag==12)return py_hold(cells[o->start+2*i+1]);
 return py_hold(cells[o->start+i*(v.tag==6?2:1)]);
}
static V get(V v,V k) {
 if(v.tag>=9)fail();
 if(v.tag==6) { int i=lookup(v,k); if(i<0)fail(); return py_hold(cells[obj(v)->start+2*i+1]); }
 return at(v,scalar(k));
}
static void put(V v,V k,V x) {
 if(v.tag==6) { store(v,k,x); return; } if(v.tag!=4)fail();
 Obj *o=obj(v); cells[o->start+norm(scalar(k),o->len)]=checked(x);
}
static int equal(V a,V b) {
 a=checked(a); b=checked(b);
 if(a.tag>=10||b.tag>=10)fail();
 if(a.tag<2&&b.tag<2)return a.x==b.x;
 if(a.tag!=b.tag)return 0; if(a.tag==2)return 1;
 if(a.tag==9)return a.x==b.x;
 if(a.tag<2)return a.x==b.x;
 Obj *x=obj(a),*y=obj(b); if(x->len!=y->len)return 0;
 if(a.tag==3) { for(int i=0;i<x->len;i++) if(chars[x->start+i]!=chars[y->start+i])return 0; return 1; }
 if(a.tag==6) {
  for(int i=0;i<x->len;i++) { int j=lookup(b,cells[x->start+2*i]);
   if(j<0||!equal(cells[x->start+2*i+1],cells[y->start+2*j+1]))return 0; } return 1;
 }
 if(a.tag==7) { for(int i=0;i<x->len;i++)if(lookup(b,cells[x->start+i])<0)return 0; return 1; }
 for(int i=0;i<x->len;i++)if(!equal(cells[x->start+i],cells[y->start+i]))return 0;
 return 1;
}
static int compare(V a,V b) {
 if(a.tag<2&&b.tag<2)return (a.x>b.x)-(a.x<b.x);
 if(a.tag!=b.tag||(a.tag!=3&&a.tag!=4&&a.tag!=5))fail();
 Obj *x=obj(a),*y=obj(b); int n=x->len<y->len?x->len:y->len;
 for(int i=0;i<n;i++) {
  int d=a.tag==3?(chars[x->start+i]>chars[y->start+i])-(chars[x->start+i]<chars[y->start+i]):
   (equal(cells[x->start+i],cells[y->start+i])?0:compare(cells[x->start+i],cells[y->start+i])); if(d)return d;
 } return (x->len>y->len)-(x->len<y->len);
}
static int contains(V v,V k) {
 if(v.tag==6||v.tag==7||v.tag==11) { V base=v;if(v.tag==11)base.tag=6;return lookup(base,k)>=0; }
 if(v.tag>=9)fail();
 Obj *o=obj(v);
 if(v.tag==3) {
  if(k.tag!=3)fail(); Obj *q=obj(k);
  for(int i=0;i<=o->len-q->len;i++) { int j=0;
   while(j<q->len&&chars[o->start+i+j]==chars[q->start+j])j++; if(j==q->len)return 1; } return 0;
 }
 for(int i=0;i<o->len;i++)if(equal(cells[o->start+i],k))return 1; return 0;
}
static V concat(V a,V b) {
 if(a.tag!=b.tag||(a.tag!=3&&a.tag!=4&&a.tag!=5))fail();
 V v=fresh(a.tag); Obj *o=obj(v),*x=obj(a),*y=obj(b); int n=x->len+y->len;
 if(a.tag==3) {
  if(n>128||ns+n>2048)fail(); o->start=ns; o->len=o->cap=n;
  for(int i=0;i<x->len;i++)chars[ns++]=chars[x->start+i];
  for(int i=0;i<y->len;i++)chars[ns++]=chars[y->start+i];
 } else { for(int i=0;i<x->len;i++)append(v,at(a,i)); for(int i=0;i<y->len;i++)append(v,at(b,i)); }
 return v;
}
static V repeat(V a,int n) {
 if(a.tag!=3&&a.tag!=4&&a.tag!=5)fail(); if(n<0)n=0; int len=length(a);
 if(len&&n>128/len)fail();if(!len)n=0; V v=fresh(a.tag); Obj *o=obj(v);
 if(a.tag==3) {
  if(ns+len*n>2048)fail(); o->start=ns; o->len=o->cap=len*n;
  for(int j=0;j<n;j++)for(int i=0;i<len;i++)chars[ns++]=chars[obj(a)->start+i];
 } else for(int j=0;j<n;j++)for(int i=0;i<len;i++)append(v,at(a,i));
 return v;
}
static V binary(int op,V a,V b) {
 a=checked(a); b=checked(b);
 if(op==0&&(a.tag>=3||b.tag>=3))return concat(a,b);
 if(op==2&&(a.tag>=3||b.tag>=3))return a.tag>=3?repeat(a,scalar(b)):repeat(b,scalar(a));
 if(op>=5) {
  if(op==5||op==6)return boolean(equal(a,b)^(op==6));
  if(op==11||op==12)return boolean(contains(b,a)^(op==12));
  if(op==13||op==14) {
   if((a.tag==0||b.tag==0)&&a.tag!=2&&b.tag!=2)fail();
   return boolean((a.tag==b.tag&&a.x==b.x)^(op==14));
  }
  int c=compare(a,b); return boolean(op==7?c<0:op==8?c<=0:op==9?c>0:c>=0);
 }
 int x=scalar(a),y=scalar(b),r=0;
 if(op==0)r=x+y; else if(op==1)r=x-y; else if(op==2)r=x*y;
 else { if(!y||(x==(-2147483647-1)&&y==-1))fail(); int q=x/y,t=x%y;
  if(t&&((t<0)!=(y<0))) { q--; t+=y; } r=op==3?q:t; }
 return num(r);
}
static V inplace(int op,V a,V b) {
 if(a.tag==4&&op==0) {
  if(b.tag!=4&&b.tag!=5)fail(); int n=length(b);
  if(length(a)+n>128)fail();for(int i=0;i<n;i++) { V x=at(b,i);append(a,x); }return a;
 }
 if(a.tag==4&&op==2) {
  int n=scalar(b),len=length(a);if(n<0||!len)n=0;if(len&&n>128/len)fail();
  if(!n)obj(a)->len=0;
  for(int j=1;j<n;j++)for(int i=0;i<len;i++) { V x=at(a,i);append(a,x); }return a;
 }
 return binary(op,a,b);
}
static V unary(int op,V a) { return op==2?boolean(!truth(a)):num(op==0?-scalar(a):scalar(a)); }
static V slice(V a,V start,V stop,V step) {
 int n=length(a),d=step.tag==2?1:scalar(step); if(!d)fail();
 int lo=start.tag==2?(d>0?0:n-1):scalar(start),hi=stop.tag==2?(d>0?n:-1):scalar(stop);
 if(start.tag!=2&&lo<0)lo+=n; if(stop.tag!=2&&hi<0)hi+=n;
 int min=d>0?0:-1,max=d>0?n:n-1;
 if(lo<min)lo=min; if(lo>max)lo=max; if(hi<min)hi=min; if(hi>max)hi=max;
 if(a.tag!=3&&a.tag!=4&&a.tag!=5)fail(); V v=fresh(a.tag); Obj *o=obj(v);
 if(a.tag==3)o->start=ns;
 for(int i=lo;d>0?i<hi:i>hi;) {
  if(a.tag==3) { if(ns==2048)fail(); chars[ns++]=chars[obj(a)->start+i]; o->len++;o->cap++; }
  else append(v,at(a,i));
  if((d>0&&i>2147483647-d)||(d<0&&i<(-2147483647-1)-d))break; i+=d;
 } return v;
}
static void outchar(int c) { printf("%c",c); }
static void repr(V v,int raw) {
 v=checked(v); if(v.tag==0) { printf("%d",v.x); return; }
 if(v.tag==1) { printf("%s",v.x?"True":"False"); return; }
 if(v.tag==2) { printf("%s","None"); return; }
 if(v.tag>=7)fail(); Obj *o=obj(v);
 if(v.tag==3) {
  int quote=39,has_single=0,has_double=0;
  for(int i=0;i<o->len;i++) { int c=chars[o->start+i]; has_single|=c==39;has_double|=c==34; }
  if(has_single&&!has_double)quote=34; if(!raw)outchar(quote);
  for(int i=0;i<o->len;i++) {
   int c=(unsigned char)chars[o->start+i];
   if(raw)outchar(c); else if(c==quote||c==92) { outchar(92);outchar(c); }
   else if(c==10||c==13||c==9) { outchar(92);outchar(c==10?'n':c==13?'r':'t'); }
   else if(c<32||c==127) {
    outchar(92);outchar('x');outchar("0123456789abcdef"[c>>4]);outchar("0123456789abcdef"[c&15]);
   } else outchar(c);
  } if(!raw)outchar(quote); return;
 }
 outchar(v.tag==4?'[':v.tag==5?'(':'{');
 for(int i=0;i<o->len;i++) {
  if(i)printf("%s",", "); repr(cells[o->start+i*(v.tag==6?2:1)],0);
  if(v.tag==6) { printf("%s",": ");repr(cells[o->start+2*i+1],0); }
 } if(v.tag==5&&o->len==1)outchar(','); outchar(v.tag==4?']':v.tag==5?')':'}');
}
