"""
let x = []; document.querySelectorAll("img").forEach(function (im){x.push(im.src)}); console.log(x)
"""

import cv2
import collections
import numpy as np
import requests
import logging
from pathlib import Path

def get_thumbnail(vid_id):
    url = f"https://img.youtube.com/vi/{vid_id}/sddefault.jpg"
    resp = requests.get(url)
    img_p = Path("imgs",f'{vid_id}.jpg')
    img_p.parent.mkdir(parents=True, exist_ok=True)
    with open(img_p,'wb') as f:
        f.write(resp.content)
    logging.info(f"Thumbnail for vid_id [{vid_id}] saved.")

def get_hist(image: np.array):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def compare_hist_correl(im1, im2):
    H1 = get_hist(im1)
    H2 = get_hist(im2)
    return cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL)

if __name__=="__main__":
    ""
    # urls = eval(open('urls.txt','r',encoding='utf-8').read())
    # for url in urls:
    #     if "hq720.jpg" not in url:
    #         continue
    #     vid_id = url.replace('https://i.ytimg.com/vi/','').split('/')[0]
    #     get_thumbnail(vid_id)

    d = collections.defaultdict(list)
    CORREL_THRESHOLD = 0.9
    imgs_dir = Path("imgs")
    for p1 in imgs_dir.glob("*.jpg"):
        for p2 in imgs_dir.glob("*.jpg"):
            if p1==p2:
                continue
            im1 = cv2.imread(str(p1))
            im2 = cv2.imread(str(p2))
            correl = compare_hist_correl(im1, im2) # [0,1]
            if correl>=CORREL_THRESHOLD:
                d[p1.stem].append((round(correl,3), p2))
        d[p1.stem].sort(reverse=True)
        print(p1, d[p1.stem][:3])