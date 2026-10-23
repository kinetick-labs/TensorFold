import json,sys,numpy as np,os
from PIL import Image
from transformers import Qwen2VLImageProcessor
opts=dict(image_mean=[.5]*3,image_std=[.5]*3,min_pixels=65536,max_pixels=16777216,patch_size=16,temporal_patch_size=2,merge_size=2)
P=Qwen2VLImageProcessor(**opts)
rng=np.random.default_rng(0)
sizes=[(480,640),(1000,750),(33,47),(512,512),(1080,1920),(300,2000),(77,1200),(2000,1500)]
caps=[(16384*1024,65536),(4096*1024,65536),(1024*256,65536)]
out=sys.argv[1]
k=0
for (h,w) in sizes:
  y,x=np.mgrid[0:h,0:w]
  img=np.stack([(x*255//max(w-1,1)),(y*255//max(h-1,1)),((x^y)&255)],-1).astype(np.uint8)
  img=np.clip(img.astype(int)+rng.integers(-40,40,img.shape),0,255).astype(np.uint8)
  for cap,mn in caps:
    r=P(images=[Image.fromarray(img)],max_pixels=cap,min_pixels=min(mn,cap))
    pv=np.asarray(r["pixel_values"],dtype=np.float32);g=np.asarray(r["image_grid_thw"])[0]
    d=f"{out}/c{k}";os.makedirs(d,exist_ok=True);k+=1
    img.tofile(d+"/rgb.u8");pv.tofile(d+"/pixels.f32")
    json.dump(dict(h=h,w=w,max_pixels=cap,min_pixels=min(mn,cap),grid=[int(v) for v in g]),open(d+"/case.json","w"))
print(k)
