# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 04:12:57 2022

@author: anjan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:31:08 2022

@author: anjan
"""

import json
import requests
from PIL import Image
from PIL import ImageColor
from io import BytesIO
import numpy as np

f=open('Bone Segments 10 images.json','r')
data = json.loads(f.read())



def url_to_mask(url):
    response = requests.get(url)
    mask_img = Image.open(BytesIO(response.content))
    mask_img=mask_img.convert("RGBA")
    return mask_img


def mask_to_color(color,bone_mask):
    color=ImageColor.getcolor(color, "RGBA")
    d_arr=np.array(bone_mask)
    new_image=np.where(d_arr>(250,250,250,0),color,d_arr)
    bone_mask=Image.fromarray(np.uint8(new_image)).convert('RGBA')
    return bone_mask


""" Commented Out Slower Code"""
# def mask_to_color(color,bone_mask):
#     d = bone_mask.getdata()
#     color=ImageColor.getcolor(color, "RGB")
#     new_image = []
#     for item in d:
#         # change all white (also shades of whites)
#         # pixels to yellow
#         if item[0] in list(range(200, 256)):
            
#             new_image.append(color)
#         else:
#             new_image.append(item)
    
             
#     # update image data
#     bone_mask.putdata(new_image)
#     return bone_mask
    
    

def paste_masks(images):
    """ 
    Input :  images   - A list of images
    
    Output : Returns a single image with all the images in the list superimposed
    """
    image0=images[0]
    for i in range(len(images)):
        image0.paste(images[i],(0,0),images[i])
    
    
    return image0

for image in range(0,10):
    img_name=data[image]['External ID']
    radio_img=Image.open('D:/Studies/MS Project/Bone Data/boneage-training-dataset/boneage-training-dataset/'+img_name)
    radio_img=radio_img.convert("RGBA")
    mask = data[image]["Label"]['objects']
    bone_url=[mask[i]['instanceURI'] for i in range(0,11)]
    bone_vals=[mask[i]['value'] for i in range(0,11)]
    colors=[mask[i]['color'] for i in range(0,11)]    
    bone_mask=[url_to_mask(bone_url[i]) for i in range(0,11)]
    bone_mask_colored=[mask_to_color(colors[i], bone_mask[i]) for i in range(0,11)]
    final_mask=paste_masks(bone_mask_colored)
    final_img_name=img_name.split('.')[0]+'_maskedx.png'
    final_mask.save(final_img_name,"PNG")
    
    print('end')


def mask_to_color(color,bone_mask):
    color=ImageColor.getcolor(color, "RGBA")
    d_arr=np.array(bone_mask)
    new_image=np.where(d_arr>(250,250,250,0),color,d_arr)
    bone_mask=Image.fromarray(np.uint8(new_image)).convert('RGBA')
    
             
    # update image data
    #bone_mask.putdata(new_image)
    return bone_mask















# img_name=data[1]['External ID']
# radio_img=Image.open('D:/Studies/MS Project/Bone Data/boneage-training-dataset/boneage-training-dataset/'+img_name)
# radio_img=radio_img.convert("RGBA")
# for mask in data[1]["Label"]['objects']:
#     color=mask['color']
#     title=mask['title']
#     value=mask['value']
#     url=mask['instanceURI']
#     print(title)
#     response = requests.get(url)
#     mask_img = Image.open(BytesIO(response.content))
#     mask_img=mask_img.convert("RGBA")
#     #mask_img.putalpha(250)
#     radio_img.paste(mask_img,(0,0),mask_img)
# final_img_name=img_name.split('.')[0]+'_masked.png'
# radio_img.save(final_img_name,"PNG")

