static V stringify(V a);
static V sort_keys[128];
static int indexable(V v) { return (v.tag>=3&&v.tag<=6)||v.tag==8||(v.tag>=10&&v.tag<=12); }
static int iterator_input(V v) { return (v.tag>=3&&v.tag<=5)||v.tag==8; }
static V iterator(int tag,int argc,V a,V b,V c) {
 if((!(tag==14&&!argc)&&!iterator_input(a))||
    (tag==14&&argc>1&&!iterator_input(b))||(tag==14&&argc>2&&!iterator_input(c)))fail();
 int n=tag==13?2:tag==15?1:argc;
 if(nc+n>1536)fail();V v=fresh(tag);Obj *o=obj(v);o->start=nc;o->len=n;o->cap=0;
 if(n)cells[nc++]=a;if(n>1)cells[nc++]=b;if(n>2)cells[nc++]=c;return v;
}
static int cg_iter(V src,int *cursor,V *out) {
 if(indexable(src)) {
  if(*cursor>=length(src))return 0;*out=at(src,(*cursor)++);return 1;
 }
 if(src.tag<13||src.tag>15)fail();Obj *o=obj(src);int i=o->cap;if(i<0)return 0;
 if(src.tag==14) {
  if(!o->len) { o->cap=-1;return 0; }
  for(int j=0;j<o->len;j++)if(i>=length(cells[o->start+j])) { o->cap=-1;return 0; }
  V pair=fresh(5);
  for(int j=0;j<o->len;j++) { V x=at(cells[o->start+j],i);append(pair,x); }
  o->cap++;*out=pair;return 1;
 }
 V input=cells[o->start];if(i>=length(input)) { o->cap=-1;return 0; }
 if(src.tag==13) {
  V pair=fresh(5);append(pair,binary(0,cells[o->start+1],num(i)));
  V x=at(input,i);append(pair,x);*out=pair;
 } else { V x=at(input,i);*out=stringify(x); }
 o->cap++;return 1;
}
static int dict_size(V v) { return v.tag==6||(v.tag>=10&&v.tag<=12)?length(v):-1; }
static void iter_guard(V v,int expected) { if(expected>=0&&length(v)!=expected)fail(); }
static V sequence(V a,int tag) {
 V v=fresh(tag),x;int cursor=0,expected=dict_size(a);
 while(1) { iter_guard(a,expected);if(!cg_iter(a,&cursor,&x))break;
  if(tag==7)setadd(v,x);else append(v,x); }return v;
}
static V total(V a) {
 V r=num(0),x;int cursor=0,expected=dict_size(a);
 while(1) { iter_guard(a,expected);if(!cg_iter(a,&cursor,&x))break;r=binary(0,r,x); }return r;
}
static V stringify(V a) {
 if(a.tag==3)return a;
 char b[12]; int n=0; unsigned int x;
 if(a.tag==2)return text("None",4);
 if(a.tag==1)return a.x?text("True",4):text("False",5);
 if(a.tag!=0)fail(); x=a.x<0?0u-(unsigned int)a.x:(unsigned int)a.x;
 do { b[n++]=(char)('0'+x%10); x/=10; }while(x);
 if(a.x<0)b[n++]='-'; V v=fresh(3); Obj *o=obj(v);
 if(ns+n>2048)fail(); o->start=ns; o->len=o->cap=n;
 while(n)chars[ns++]=b[--n]; return v;
}
static int integer(V a) {
 if(a.tag<2)return scalar(a); if(a.tag!=3)fail(); Obj *o=obj(a); int i=0,neg=0,n=0;
 while(i<o->len&&(chars[o->start+i]==' '||chars[o->start+i]=='\t'||chars[o->start+i]=='\n'))i++;
 if(i<o->len&&(chars[o->start+i]=='-'||chars[o->start+i]=='+'))neg=chars[o->start+i++]=='-';
 int digits=0;
 while(i<o->len&&chars[o->start+i]>='0'&&chars[o->start+i]<='9') {
  int d=chars[o->start+i++]-'0'; if(n<(-2147483647-1)/10)fail();
  n*=10; if(n<(-2147483647-1)+d)fail(); n-=d; digits++;
 }
 while(i<o->len&&(chars[o->start+i]==' '||chars[o->start+i]=='\t'||chars[o->start+i]=='\n'))i++;
 if(!digits||i!=o->len||(!neg&&n==(-2147483647-1)))fail(); return neg?n:-n;
}
static V builtin(int id,int argc,V a,V b,V c) {
 static const unsigned char minargs[]={1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,2};
 static const unsigned char maxargs[]={1,2,3,3,3,1,1,1,1,2,3,3,1,1,1,1,1,1,2};
 if(id<0||id>18||argc<minargs[id]||argc>maxargs[id])fail();
 if(id==0)return num(length(a));
 if(id==1) {
  V r=argc>1?b:num(0),x;int cursor=0,expected=dict_size(a);
  scalar(r);while(1) { iter_guard(a,expected);if(!cg_iter(a,&cursor,&x))break;
   scalar(x);r=binary(0,r,x); }return r;
 }
 if(id==2||id==3) {
  if(argc>1) {
   V r=a;int d=compare(b,r);if(id==2?d<0:d>0)r=b;
   if(argc>2) { d=compare(c,r);if(id==2?d<0:d>0)r=c; }return r;
  }
  V r,x;int cursor=0,expected=dict_size(a);if(!cg_iter(a,&cursor,&r))fail();
  while(1) { iter_guard(a,expected);if(!cg_iter(a,&cursor,&x))break;
   int d=compare(x,r);if(id==2?d<0:d>0)r=x; }return r;
 }
 if(id==4) {
  int lo=argc==1?0:scalar(a),hi=argc==1?scalar(a):scalar(b),step=argc==3?scalar(c):1;
  if(!step)fail(); V v=fresh(8);
  for(int i=lo;step>0?i<hi:i>hi;) {
   append(v,num(i)); if((step>0&&i>2147483647-step)||(step<0&&i<(-2147483647-1)-step))break; i+=step;
  } return v;
 }
 if(id>=5&&id<=7)return argc?sequence(a,id==5?4:id==6?5:7):fresh(id==5?4:id==6?5:7);
 if(id==8) {
  V v=fresh(6); if(!argc)return v;
  if(a.tag==6) { Obj *o=obj(a); for(int i=0;i<o->len;i++)store(v,cells[o->start+2*i],cells[o->start+2*i+1]); }
  else { V pair;int cursor=0,expected=dict_size(a);
   while(1) { iter_guard(a,expected);if(!cg_iter(a,&cursor,&pair))break;
    if(length(pair)!=2)fail();V k=at(pair,0),x=at(pair,1);store(v,k,x); }
  }
  return v;
 }
 if(id==9)return iterator(13,2,a,argc>1?b:num(0),none());
 if(id==10)return iterator(14,argc,a,b,c);
 if(id==11) {
  V v;
  if(a.tag==7) { v=fresh(4);for(int i=0;i<length(a);i++)append(v,py_hold(cells[obj(a)->start+i])); }
  else v=sequence(a,4);
  if(b.tag!=2&&(b.tag!=9||b.x!=1))fail();
  int key=b.tag==9?b.x:0,rev=c.tag==2?0:scalar(c)!=0; Obj *o=obj(v);
  for(int i=0;i<o->len;i++) {
   V x=py_hold(cells[o->start+i]);sort_keys[i]=key?total(x):x;
  }
  for(int i=1;i<o->len;i++) {
   V x=py_hold(cells[o->start+i]),kx=py_hold(sort_keys[i]); int j=i;
   while(j>0) { V ky=py_hold(sort_keys[j-1]); int d=compare(ky,kx);
    if(rev?d>=0:d<=0)break;
    cells[o->start+j]=py_hold(cells[o->start+j-1]);sort_keys[j]=ky;j--; }
   cells[o->start+j]=x;sort_keys[j]=kx;
  } return v;
 }
 if(id==12)return argc?stringify(a):emptytext();
 if(id==13)return num(argc?integer(a):0);
 if(id==14)return boolean(argc?truth(a):0);
 if(id==15) { int n=scalar(a);if(n==(-2147483647-1))fail();return num(n<0?-n:n); }
 if(id==16||id==17) {
  V x;int cursor=0,expected=dict_size(a);
  while(1) { iter_guard(a,expected);if(!cg_iter(a,&cursor,&x))break;
   if(truth(x)==(id==16))return boolean(id==16); }return boolean(id==17);
 }
 if(id==18) { if(argc!=2||a.tag!=9||a.x!=2)fail();return iterator(15,1,b,none(),none()); }

 fail();
}
static V method(int id,V a,int argc,V b,V c,V d) {
 static const unsigned char minargs[]={1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0};
 static const unsigned char maxargs[]={1,1,0,2,2,0,0,0,1,1,1,3,3,1,1,0};
 if(id<0||id>15||argc<minargs[id]||argc>maxargs[id])fail();
 if(id==0) { if(a.tag!=4)fail();append(a,b);return none(); }
 if(id==1) {
  if(a.tag!=4)fail();Obj *o=obj(a);int i=norm(argc?scalar(b):-1,o->len);V x=cells[o->start+i];
  for(int j=i+1;j<o->len;j++)cells[o->start+j-1]=cells[o->start+j];o->len--;return x;
 }
 if(id==2) {
  if(a.tag!=4)fail();Obj *o=obj(a);for(int i=0;i<o->len/2;i++) {
   V x=cells[o->start+i];cells[o->start+i]=cells[o->start+o->len-i-1];cells[o->start+o->len-i-1]=x;
  }return none();
 }
 if(id==3||id==4) {
  if(a.tag!=6)fail();int i=lookup(a,b);if(i>=0)return py_hold(cells[obj(a)->start+2*i+1]);
  V x=argc>1?c:none();if(id==4)store(a,b,x);return x;
 }
 if(id>=5&&id<=7) {
  if(a.tag!=6)fail();return val(id==5?11:id==6?10:12,a.x);
 }
 if(id==8) { setadd(a,b);return none(); }
 if(id==9||id==10) {
  if(a.tag!=7)fail();int i=lookup(a,b);if(i<0) { if(id==9)fail();return none(); }
  Obj *o=obj(a);for(int j=i+1;j<o->len;j++)cells[o->start+j-1]=cells[o->start+j];o->len--;return none();
 }
 if(id==11||id==12) {
  if(a.tag!=3||b.tag!=3)fail();Obj *o=obj(a),*q=obj(b);int lo=argc>1?scalar(c):0,hi=argc>2?scalar(d):o->len;
  if(lo<0)lo+=o->len;if(lo<0)lo=0;if(hi<0)hi+=o->len;if(hi<0)hi=0;if(hi>o->len)hi=o->len;
  int found=-1;
  for(int i=lo;i<=hi-q->len;i++) { int j=0;while(j<q->len&&chars[o->start+i+j]==chars[q->start+j])j++;
   if(j==q->len) { found=i;if(id==11)break; } }return num(found);
 }
 if(id==13) {
  if(a.tag!=3)fail();V seq=sequence(b,4);int n=length(seq),size=0;
  for(int i=0;i<n;i++) { V x=at(seq,i);if(x.tag!=3)fail();size+=length(x)+(i?length(a):0); }
  if(size>128||ns+size>2048)fail();V v=fresh(3);Obj *o=obj(v);o->start=ns;o->len=o->cap=size;
  for(int i=0;i<n;i++) { if(i)for(int j=0;j<length(a);j++)chars[ns++]=chars[obj(a)->start+j];
   V x=at(seq,i);for(int j=0;j<length(x);j++)chars[ns++]=chars[obj(x)->start+j]; }return v;
 }
 if(id==14) {
  if(a.tag!=3||argc!=1||b.tag!=3||!length(b))fail();V v=fresh(4);Obj *o=obj(a),*q=obj(b);int start=0;
  for(int i=0;i<=o->len-q->len;i++) { int j=0;while(j<q->len&&chars[o->start+i+j]==chars[q->start+j])j++;
   if(j==q->len) { V x=text((volatile char *)(chars+o->start+start),i-start);append(v,x);i+=q->len-1;start=i+1; }
  }V x=text((volatile char *)(chars+o->start+start),o->len-start);append(v,x);return v;
 }
 if(id==15) { if(a.tag!=4)fail();return sequence(a,4); }
 fail();
}
