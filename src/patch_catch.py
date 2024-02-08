from xml.dom import minidom
import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import opening,closing
from skimage import measure,morphology
from recognition import recognition
import time
import warnings
import os
import shutil
import xlwt
import time
from ultralytics import YOLO

warnings.filterwarnings("ignore")

class patch_catch():
    def __init__(self, templet_xml_path, img, seg):
        # templet_xml_path：xml模板
        # img: 图像
        # seg: 图像的框线分割结果
        self.doc = minidom.parse(templet_xml_path)
        self.img = img  # h*w*c
        self.seg = seg  # h*w*c


#---------------------------tool-------------------------#
          
    # 获取节点值
    def getNodeText(self,node):
        nodelist = node.childNodes
        result = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                result.append(node.data)
        return ''.join(result)

    # 判断矩形四个顶点，左上、右下、右上、左下
    def findApex(self,loc):
        loc_sum = np.sum(loc, axis=1)
        ind = np.argsort(loc_sum)
        left_top = loc[ind[0], :]
        right_bottom = loc[ind[-1], :]

        loc_diff = np.diff(loc).squeeze()
        ind = np.argsort(loc_diff)
        right_top = loc[ind[-1], :]
        left_bottom = loc[ind[0], :]

        return left_top,right_bottom,right_top,left_bottom

    # 删除二值图像中的白色小噪点
    def baweraopen(self,image, size):
        img_contour = image.copy()
        contours, hierarchy = cv.findContours(img_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i in contours:
            res = cv.contourArea(i)
            if res < size:
                cv.fillConvexPoly(image, cv.convexHull(i), 0)
        return image

     # 删除二值图像中的白色小噪点
    def baweraopen_seg(self,image, size):

        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            x, y, w, h = cv.boundingRect(contours[i])
            res = w*h
            if res < size:
                cv.fillConvexPoly(image, contours[i], 0)
        return image

    def get_rowList(self,img_bgr):
        

        gray = cv.cvtColor(img_bgr, cv.COLOR_GRAY2BGR)
        img_gray=cv.cvtColor(gray,cv.COLOR_BGR2GRAY)
        #二值化
        t,binary=cv.threshold(img_gray,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY)
        # t,binary=cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY)
        '''
        水平投影从左向右投射，计算每一行的黑色像素总数
        '''
        rows,cols=binary.shape
        hor_list=[0]*rows
        for i in range(rows):
            for j in range(cols):
                #统计每一行的黑色像素总数
                if binary.item(i,j)==0:
                    hor_list[i]=hor_list[i]+1
        '''
        对hor_list中的元素进行筛选，可以去除一些噪点
        '''
        hor_arr=np.array(hor_list)
        hor_arr[np.where(hor_arr<5)]=0
        hor_list=hor_arr.tolist()

        #取出各个文字区间
        vv_list=self.get_vvList(hor_list)
        
            # cv.imshow('文本行',img_hor)
            # cv.waitKey(0)
        return vv_list

    def get_vvList(self,list_data):
        #取出list中像素存在的区间
        vv_list=list()
        v_list=list()
        for index,i in enumerate(list_data):
            if i>0:
                v_list.append(index)
            else:
                if v_list:
                    vv_list.append(v_list)
                    #list的clear与[]有区别
                    v_list=[]
        return vv_list

    # 删除二值图像中的多余白边
    def delNoneZero(self,image):
        img_contour = image.copy()
        gray = cv.cvtColor(img_contour, cv.COLOR_GRAY2BGR)
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

        gray = 255*(gray < 128).astype(np.uint8)
        coords = cv.findNonZero(gray)
        x, y, w, h = cv.boundingRect(coords)
        padding = 0
        rect = image[y-padding:y+h+padding, x-padding:x+w+padding]
        return rect

    
    # 填充二值图像闭合空洞区域
    def FillHole(self,mask):
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out
#---------------------------tool-------------------------#

# --------------------------获取模板中的交点标注-----------------------#
    def get_template_corner(self):

        #提取模板尺寸
        img_h = int(self.getNodeText(self.doc.getElementsByTagName("pagelength")[0]))
        img_w = int(self.getNodeText(self.doc.getElementsByTagName("pagewidth")[0]))

        # 提取数字roi位置
        numLocs = self.doc.getElementsByTagName("numLoc")
        number_id = []
        for numLoc in numLocs:
            temp = []
            loc = self.getNodeText(numLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            number_id.append(temp)


        # 提取ocrroi位置
        ocrLocs = self.doc.getElementsByTagName("ocrLoc")
        ocr_id = []
        for ocrLoc in ocrLocs:
            temp = []
            loc = self.getNodeText(ocrLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            ocr_id.append(temp)
        
        # 提取mulitnumroi位置
        mulitnumLocs = self.doc.getElementsByTagName("mulitnumLoc")
        mulitnum_id = []
        for mulitnumLoc in mulitnumLocs:
            temp = []
            loc = self.getNodeText(mulitnumLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            mulitnum_id.append(temp)

         # 提取writetextroi位置
        writetextLocs = self.doc.getElementsByTagName("writetextLoc")
        writetext_id = []
        for writetextLoc in writetextLocs:
            temp = []
            loc = self.getNodeText(writetextLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            writetext_id.append(temp)

         # 新提取writetextcheckroi位置
        writetextcheckLocs = self.doc.getElementsByTagName("writetextcheckLoc")
        writetextcheck_id = []
        for writetextcheckLoc in writetextcheckLocs:
            temp = []
            loc = self.getNodeText(writetextcheckLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            writetextcheck_id.append(temp)

        # 提取mixtextroi位置
        mixtextLocs = self.doc.getElementsByTagName("mixtextLoc")
        mixtext_id = []
        mixtext_type=[]
        mixtext_loc=[]
        mixtext_rowloc=[]
        for mixtextLoc in mixtextLocs:
            temp = []
            loc = self.getNodeText(mixtextLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            mixtext_id.append(temp)
 
            temp = []
            loc = self.getNodeText(mixtextLoc.getElementsByTagName("columnType")[0])
            loc = loc.split('|')
            for i in loc:
                temp.append(int(i))
            mixtext_type.append(temp)
         
            temp = []
            loc = self.getNodeText(mixtextLoc.getElementsByTagName("columnLoc")[0])
            loc = loc.split('|')
            for i in loc:
                temp.append(int(i))
            mixtext_loc.append(temp)

            temp = []
            loc = self.getNodeText(mixtextLoc.getElementsByTagName("rowLoc")[0])
            loc = loc.split('|')
            for i in loc:
                temp.append(int(i))
            mixtext_rowloc.append(temp)
        


        #提取方形roi位置
        ckbLocs = self.doc.getElementsByTagName("ckbLoc")
        checkbox_id = []
        for ckbLoc in ckbLocs:
            temp = []
            loc = self.getNodeText(ckbLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            checkbox_id.append(temp)
        # 提取圆形roi位置
        cirLocs = self.doc.getElementsByTagName("cirLoc")
        cirbox_id = []
        for cirLoc in cirLocs:
            temp = []
            loc = self.getNodeText(cirLoc.getElementsByTagName("loc")[0])
            loc = loc.split(' ')
            for i in loc:
                temp.append(int(i))
            cirbox_id.append(temp)

        #提取框线每个交点
        tables = self.doc.getElementsByTagName("table")

        templet_corner = []
        templet_num = []
        for table in tables:
            h = self.getNodeText(table.getElementsByTagName("textTop")[0])
            w = self.getNodeText(table.getElementsByTagName("textLeft")[0])
            name = self.getNodeText(table.getElementsByTagName("name")[0])
            templet_corner.append([int(h),int(w)]) #统一为A4大小
            templet_num.append(int(name[2:]))

        templet_corner = np.array(templet_corner)

        # 判断表单框线的四个顶点
        templet_left_top,templet_right_bottom,templet_right_top,templet_left_bottom = self.findApex(templet_corner)

        templet_vertex = [templet_left_top, templet_left_bottom,templet_right_bottom]

        return templet_corner,templet_num,templet_vertex,number_id,checkbox_id, cirbox_id,ocr_id,mulitnum_id,writetext_id,writetextcheck_id,mixtext_id,mixtext_type,mixtext_loc,mixtext_rowloc, img_h,img_w

# --------------------------获取模板中的交点标注-----------------------#

#------------------------------仿射纠偏---------------------------#
    def affine_correction(self,img,seg,templet_vertex,img_h,img_w):

        #输入的分割结果为白底黑边
        seg = ~seg
        seg = self.baweraopen(seg,5)
        seg = ~seg
        seg = np.expand_dims(seg, axis=2)
        seg = np.concatenate((seg, seg, seg), axis=-1)

        img = cv.resize(img, (img_w,img_h),interpolation=cv.INTER_LINEAR) #Size(width，height)
        seg = cv.resize(seg, (img_w,img_h),interpolation=cv.INTER_LINEAR) #Size(width，height)

        _, seg = cv.threshold(seg, 254, 255, cv.THRESH_BINARY)

        seg_b = ~seg[:,:,1]
        
        k = np.ones((3, 3), np.uint8)
        seg_b = cv.morphologyEx(seg_b, cv.MORPH_CLOSE, k,1)
        seg_b = self.FillHole(seg_b.astype('uint8'))     
        k = np.ones((10, 10), np.uint8)   
        seg_b = cv.morphologyEx(seg_b, cv.MORPH_OPEN, k,1)
        _, seg_b = cv.threshold(seg_b, 0, 255, cv.THRESH_BINARY)


        seg_b = self.baweraopen(seg_b.astype('uint8'),seg_b.shape[0]*seg_b.shape[1]/20)
        

        [x,y] = np.where(seg_b==255)
        loc = np.vstack((x,y)).transpose([1,0])


        loc = np.array(loc)      

        x1,y1,w1,h1 = cv.boundingRect(loc)
        left_top1=[x1,y1]
        right_bottom1=[x1+w1,y1+h1]
        left_bottom1=[x1+w1,y1]
        right_top1=[x1,y1+h1]

        left_top,right_bottom,right_top,left_bottom = self.findApex(loc)

        y_sense=20

        if (left_top[0]>right_top[0]) and ((left_top[0]-right_top[0]))>y_sense:
            left_top[0]=right_top[0]


        src = np.float32([left_top,left_bottom,right_bottom])
        dst = np.float32([left_top1,left_bottom1,right_bottom1])
        # dst = np.float32(templet_vertex)

        ###仿射变换###
        A1 = cv.getAffineTransform(src[:,[1,0]],dst[:,[1,0]])#输入需要(w,h)
        rectify_seg = cv.warpAffine(seg,A1, (img_w,img_h), borderValue=(255,255,255))
        rectify_img = cv.warpAffine(img,A1, (img_w,img_h), borderValue=(255,255,255))

        _, rectify_seg = cv.threshold(rectify_seg, 254, 255, cv.THRESH_BINARY)


        return rectify_img,rectify_seg
#------------------------------仿射纠偏---------------------------#

#--------------------------分割结果交点检测-------------------------#
    def seg_corner_detection(self,rectify_seg,imgname):
        seg_b = np.float32(~rectify_seg[:,:,1])
        seg_b[seg_b==255] = 1

      
        #--------角点检测方法3构造滤波器------#
        skel_show = seg_b
        colormap = np.zeros(rectify_seg.shape)

        kernel1 = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        row = opening(skel_show, kernel1)

        colormap[row==1,:] = 255


        kernel3 = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
        col = opening(skel_show, kernel3)

        colormap[col == 1, :] = 255


        inter = row * col

        labels = measure.label(inter)
        propsa = measure.regionprops(labels)

        seg_corner = []
        for area in propsa:
            corner_loc = [round(area.centroid[0]), round(area.centroid[1])]
            seg_corner.append(corner_loc)
        #--------角点检测方法3构造滤波器------#

        temp_img = rectify_seg.copy()
        for i in seg_corner:
            cv.circle(colormap, (int(i[1]),int(i[0])), 7, (0, 0, 255), -1)

        return seg_corner
#--------------------------分割结果交点检测-------------------------#

#-----------------------------交点校验----------------------------#
    def corner_check(self,seg_corner,templet_corner,templet_num,rectify_img,imgname):

        seg_corner = np.array(seg_corner)

        temp_seg_corner = seg_corner.copy()
        true_corner = []
        true_num = []
        for i,loc_i in enumerate(templet_corner):
            temp_loc = []
            temp_distance = []
            for j,loc_j in enumerate(temp_seg_corner):
                distance = abs(loc_i[0]-loc_j[0]) + abs(loc_i[1]-loc_j[1]) #曼哈顿距离
                if distance <= 15:  #if distance <= 10:
                    temp_loc.append(j)
                    temp_distance.append(distance)
            if len(temp_loc) == 0:
                true_corner.append(loc_i)
                true_num.append(templet_num[i])
            elif len(temp_loc) == 1:
                true_corner.append(temp_seg_corner[temp_loc[0],:])
                true_num.append(templet_num[i])
                temp_seg_corner = np.delete(temp_seg_corner, temp_loc[0], axis=0)
            else:
                ind = np.argsort(np.array(temp_distance))
                true_corner.append(temp_seg_corner[temp_loc[ind[0]], :])
                true_num.append(templet_num[i])
                temp_seg_corner = np.delete(temp_seg_corner, temp_loc[ind[0]], axis=0)

        temp_img = np.copy(rectify_img)
        for i in true_corner:
            cv.circle(temp_img, (int(i[1]),int(i[0])), 7, (0, 255, 0), -1)
            cv.circle(temp_img, (int(i[1]),int(i[0])), 10, (0, 255, 0), -1)

        return true_corner, true_num
#-----------------------------交点校验----------------------------#

#------------------------根据交点提取目标区域-----------------------#

    # 对扣取的ocr patch进行去框
    def patch_ocr_preprocessing(self,img):
        gray_img = img[:, :, 0]

        return gray_img

    # 对扣取的小patch进行去框
    def patch_preprocessing(self,img):
        gray_img = img[:, :, 0]

        return gray_img

    # 根据四个交点从图像中扣取小patch，对偏斜的目标进行微调校正
    def extract_roi_mix(self,true_corner,rectify_img,cropped_begin,cropped_rowbegin,loc,rowloc,type,roi,temid):

        true_corner = np.array(true_corner)
        roi_corner = true_corner[roi,:] #(h*w)

        roi_left_top,roi_right_bottom,roi_right_top,roi_left_bottom = self.findApex(roi_corner)
         
        cropped = rectify_img[roi_left_top[0]+5:roi_right_bottom[0]-5, roi_left_top[1]+5:roi_right_bottom[1]-5]


        # 按比例切割图片
        w=cropped.shape[1]
        h=cropped.shape[0]
        col_start=round(h*cropped_rowbegin/10000)
        col_end=round(h*rowloc/10000)
        row_start=round(w*cropped_begin/10000)
        row_end=round(w*loc/10000)

        cropped = cropped[col_start:col_start+col_end, row_start:row_start+row_end]

        cropped = self.patch_ocr_preprocessing(cropped)

  
        return cropped

    # 根据四个交点从图像中扣取小patch，对偏斜的目标进行微调校正
    def extract_roi(self,true_corner,rectify_img,type,roi):

        true_corner = np.array(true_corner)
        roi_corner = true_corner[roi,:] #(h*w)

        roi_left_top,roi_right_bottom,roi_right_top,roi_left_bottom = self.findApex(roi_corner)
        if type == 'num' :
         
            step = 2
            cropped = rectify_img[roi_left_top[0]+step:roi_right_bottom[0]-step, roi_left_top[1]+step:roi_right_bottom[1]-step]
            if cropped.size==0:

                cropped = rectify_img[roi_left_top[0]:roi_right_bottom[0], roi_left_top[1]:roi_right_bottom[1]]
            else:
                cropped = self.patch_preprocessing(cropped)
                cropped = select_max_region(cropped)   #add by yangt

            #平移校正
            if np.where(cropped == 0)[0].size != 0:
                for i in range(3):
                    cropped1 = np.expand_dims(cropped, axis=2)
                    cropped1 = np.concatenate((cropped1, cropped1, cropped1), axis=-1)
                    h, w, c = cropped1.shape
                    cropped_t = ~cropped1[:, :, 0]
                    sum_col = np.sum(cropped_t, axis=0)#按列求和
                    sum_row = np.sum(cropped_t, axis=1)  # 按行求和
                    site_w = np.where(sum_col != 0)
                    site_h = np.where(sum_row != 0)
                    h_mean,w_mean = np.mean(site_h[0]),np.mean(site_w[0])
                    if np.isnan(h_mean) or np.isnan(w_mean):
                        break
                    else:
                        h_shift,w_shift = round(h_mean - h/2), round(w_mean - w/2)
                        matShift2 = np.float32([[1, 0, -w_shift], [0, 1, -h_shift]])
                        cropped1 = cv.warpAffine(cropped1, matShift2, (w, h), borderValue=(255, 255, 255))
                        cropped = self.patch_preprocessing(cropped1)

        elif type=='ocr' :
            step = 2
            cropped = rectify_img[roi_left_top[0]+step:roi_right_bottom[0]-step, roi_left_top[1]+step:roi_right_bottom[1]-step]
            cropped = self.patch_preprocessing(cropped)


        elif   type=='mulitnum' or type=='writetext' or type=='writetextcheck':
            cropped = rectify_img[roi_left_top[0]:roi_right_bottom[0], roi_left_top[1]:roi_right_bottom[1]]  
            cropped = self.patch_preprocessing(cropped)
 

        else:
            loc = np.array(roi_corner)      
            x1,y1,w1,h1 = cv.boundingRect(loc)
            roi_left_top=[x1,y1]
            roi_right_bottom=[x1+w1,y1+h1]
            roi_left_bottom=[x1+w1,y1]
            roi_right_top=[x1,y1+h1]
            if type == 'cir':  
                
                cropped = rectify_img[roi_left_top[0]-5 :roi_right_bottom[0]+5 ,roi_left_top[1]-5 :roi_right_bottom[1]+5 , :]   
                cropped = cropped[:, :, 1]
                cropped = select_max_region(cropped)   
                
            else:
                #平移校正
                h_shift,w_shift = float('inf'),float('inf')
                cum_h, cum_w = 0, 0
                while abs(h_shift) + abs(w_shift) >= 0.005:
                    cropped = rectify_img[roi_left_top[0]+cum_h - 5:roi_right_bottom[0]+cum_h + 5,roi_left_top[1]+cum_w - 5:roi_right_bottom[1]+cum_w + 5,:]
                    h,w,c = cropped.shape
                    cropped_t = ~cropped[:, :, 0]
                    sum_col = np.sum(cropped_t, axis=0)#按列求和
                    sum_row = np.sum(cropped_t, axis=1)  # 按行求和
                    site_w = np.where(sum_col != 0)
                    site_h = np.where(sum_row != 0)
                    h_mean,w_mean = np.mean(site_h[0]),np.mean(site_w[0])
                    if np.isnan(h_mean) or np.isnan(w_mean):
                        break
                    else:
                        h_shift,w_shift = round(h_mean - h/2), round(w_mean - w/2)
                        cum_h += h_shift
                        cum_w += w_shift

        return cropped

    # 将每个patch图像块进行排列，形成具有实际意义的字符串
    def sort_string(self, templet_corner, number_id, checkbox_id, cirbox_id,ocr_id,mulitnum_id,writetext_id,writetextcheck_id,mixtext_id,mixtext_type,mixtext_rowloc):

        index_num = []
        index_ckb = []
        index_cir = []
        index_ocr = []
        index_mulitnum = []
        index_writetext= []
        index_writetextcheck= []
        index_mixtext= []

        # 数字
        if len(number_id) != 0:
            num_center = []
            for n, p in enumerate(number_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                num_center.append(temp)
            num_center = np.array(num_center)

            thr = roi_right_bottom[1] - roi_left_top[1]

            # 按行拆分
            temp_num_center = num_center.copy()
            temp_num_center = np.insert(temp_num_center,0,range(0,temp_num_center.shape[0]),axis=1)
            h_sort = num_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = num_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_num_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_num_center[temp_loc,:]
                row_loc = np.array(row_loc)
                row_loc = row_loc[row[:,2].argsort()]
                index.append(list(row_loc))
                temp_num_center = np.delete(temp_num_center, temp_loc, axis=0)

            index = list(filter(None, index))

            # 进一步按列拆分
            index_new = []
            for i in index:
                weight = num_center[i, 1]
                diff2_w = np.diff(weight)
                t = np.where(diff2_w >= thr*1.2)[0]
                if t.size == 0:
                    index_new.append(i)
                else:
                    t = t + 1
                    t = np.insert(np.append(t, len(i)), 0, 0)
                    for j in range(len(t)-1):
                        index_new.append(i[t[j]:t[j+1]])
            index_num = index_new


        # ocr
        if len(ocr_id) != 0:
            num_center = []
            for n, p in enumerate(ocr_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                num_center.append(temp)
            num_center = np.array(num_center)

            thr = roi_right_bottom[1] - roi_left_top[1]

            # 按行拆分
            temp_num_center = num_center.copy()
            temp_num_center = np.insert(temp_num_center,0,range(0,temp_num_center.shape[0]),axis=1)
            h_sort = num_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = num_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_num_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_num_center[temp_loc,:]
                row_loc = np.array(row_loc)
                # print(row_loc)
                # print(num_center[row_loc, :])
                row_loc = row_loc[row[:,2].argsort()]
                # print(num_center[row_loc,:])
                index.append(list(row_loc))
                temp_num_center = np.delete(temp_num_center, temp_loc, axis=0)
                # print(h_sort.size)

            index = list(filter(None, index))

            # 进一步按列拆分
            index_new = []
            for i in index:
                weight = num_center[i, 1]
                diff2_w = np.diff(weight)
                t = np.where(diff2_w >= thr*1.2)[0]
                if t.size == 0:
                    index_new.append(i)
                else:
                    t = t + 1
                    t = np.insert(np.append(t, len(i)), 0, 0)
                    for j in range(len(t)-1):
                        index_new.append(i[t[j]:t[j+1]])
                        # print(index_new)
            index_ocr = index_new

        # mulitnum
        if len(mulitnum_id) != 0:
            num_center = []
            for n, p in enumerate(mulitnum_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                num_center.append(temp)
            num_center = np.array(num_center)

            thr = roi_right_bottom[1] - roi_left_top[1]

            # 按行拆分
            temp_num_center = num_center.copy()
            temp_num_center = np.insert(temp_num_center,0,range(0,temp_num_center.shape[0]),axis=1)
            h_sort = num_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = num_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_num_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_num_center[temp_loc,:]
                row_loc = np.array(row_loc)
                # print(row_loc)
                # print(num_center[row_loc, :])
                row_loc = row_loc[row[:,2].argsort()]
                # print(num_center[row_loc,:])
                index.append(list(row_loc))
                temp_num_center = np.delete(temp_num_center, temp_loc, axis=0)
                # print(h_sort.size)

            index = list(filter(None, index))

            # 进一步按列拆分
            index_new = []
            for i in index:
                weight = num_center[i, 1]
                diff2_w = np.diff(weight)
                t = np.where(diff2_w >= thr*1.2)[0]
                if t.size == 0:
                    index_new.append(i)
                else:
                    t = t + 1
                    t = np.insert(np.append(t, len(i)), 0, 0)
                    for j in range(len(t)-1):
                        index_new.append(i[t[j]:t[j+1]])
                        # print(index_new)
            index_mulitnum = index_new

        # writetext
        if len(writetext_id) != 0:
            num_center = []
            for n, p in enumerate(writetext_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                num_center.append(temp)
            num_center = np.array(num_center)

            thr = roi_right_bottom[1] - roi_left_top[1]

            # 按行拆分
            temp_num_center = num_center.copy()
            temp_num_center = np.insert(temp_num_center,0,range(0,temp_num_center.shape[0]),axis=1)
            h_sort = num_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = num_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_num_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_num_center[temp_loc,:]
                row_loc = np.array(row_loc)
                # print(row_loc)
                # print(num_center[row_loc, :])
                row_loc = row_loc[row[:,2].argsort()]
                # print(num_center[row_loc,:])
                index.append(list(row_loc))
                temp_num_center = np.delete(temp_num_center, temp_loc, axis=0)
                # print(h_sort.size)

            index = list(filter(None, index))

            # 进一步按列拆分
            index_new = []
            for i in index:
                weight = num_center[i, 1]
                diff2_w = np.diff(weight)
                t = np.where(diff2_w >= thr*1.2)[0]
                if t.size == 0:
                    index_new.append(i)
                else:
                    t = t + 1
                    t = np.insert(np.append(t, len(i)), 0, 0)
                    for j in range(len(t)-1):
                        index_new.append(i[t[j]:t[j+1]])
                        # print(index_new)
            index_writetext = index_new

            # 按columnLoc进行拆分
            # 按columnLoc进行拆分

        # writetextcheck
        if len(writetextcheck_id) != 0:
            num_center = []
            for n, p in enumerate(writetextcheck_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                num_center.append(temp)
            num_center = np.array(num_center)

            thr = roi_right_bottom[1] - roi_left_top[1]

            # 按行拆分
            temp_num_center = num_center.copy()
            temp_num_center = np.insert(temp_num_center,0,range(0,temp_num_center.shape[0]),axis=1)
            h_sort = num_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = num_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_num_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_num_center[temp_loc,:]
                row_loc = np.array(row_loc)
                # print(row_loc)
                # print(num_center[row_loc, :])
                row_loc = row_loc[row[:,2].argsort()]
                # print(num_center[row_loc,:])
                index.append(list(row_loc))
                temp_num_center = np.delete(temp_num_center, temp_loc, axis=0)
                # print(h_sort.size)

            index = list(filter(None, index))

            # 进一步按列拆分
            index_new = []
            for i in index:
                weight = num_center[i, 1]
                diff2_w = np.diff(weight)
                t = np.where(diff2_w >= thr*1.2)[0]
                if t.size == 0:
                    index_new.append(i)
                else:
                    t = t + 1
                    t = np.insert(np.append(t, len(i)), 0, 0)
                    for j in range(len(t)-1):
                        index_new.append(i[t[j]:t[j+1]])
                        # print(index_new)
            index_writetextcheck = index_new


        # mixtext
        if len(mixtext_id) != 0:
            num_center = []
            for n, p in enumerate(mixtext_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                num_center.append(temp)
            num_center = np.array(num_center)

            thr = roi_right_bottom[1] - roi_left_top[1]

            # 按行拆分
            temp_num_center = num_center.copy()
            temp_num_center = np.insert(temp_num_center,0,range(0,temp_num_center.shape[0]),axis=1)
            h_sort = num_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = num_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_num_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_num_center[temp_loc,:]
                row_loc = np.array(row_loc)
                # print(row_loc)
                # print(num_center[row_loc, :])
                row_loc = row_loc[row[:,2].argsort()]
                # print(num_center[row_loc,:])
                index.append(list(row_loc))
                temp_num_center = np.delete(temp_num_center, temp_loc, axis=0)
                # print(h_sort.size)

            index = list(filter(None, index))

            # 进一步按列拆分
            index_new = []
            for i in index:
                weight = num_center[i, 1]
                diff2_w = np.diff(weight)
                t = np.where(diff2_w >= thr*1.2)[0]
                if t.size == 0:
                    index_new.append(i)
                else:
                    t = t + 1
                    t = np.insert(np.append(t, len(i)), 0, 0)
                    for j in range(len(t)-1):
                        index_new.append(i[t[j]:t[j+1]])
                        # print(index_new)
            index_mixtext = index_new
            current_num=0
            if len(index_num)>0:
                current_num=max(index_num)[0]+1
            current_ocr=0
            if len(index_ocr)>0:
                current_ocr=max(index_ocr)[0]+1
            current_mulitnum=0
            if len(index_mulitnum)>0:
                current_mulitnum=max(index_mulitnum)[0]+1
            current_writetext=0
            if len(index_writetext)>0:
                current_writetext=max(index_writetext)[0]+1
            if len(index_writetextcheck)>0:
                current_writetextcheck=max(index_writetextcheck)[0]+1    

            # 加入行数的计算
               
            for m, y in enumerate(mixtext_type):
                for a, b in enumerate(mixtext_rowloc[m]):
                    for x, r in enumerate(y):
                        # 1、手写单数,4、印刷体ocr，7、手写多数字，8、手写汉字
                        
                        if  r==1: 
                            
                            index_num.append([current_num])
                            current_num=current_num+1
                        elif r==4:
                            index_ocr.append([current_ocr])
                            current_ocr=current_ocr+1
                        elif r==7:
                            index_mulitnum.append([current_mulitnum])
                            current_mulitnum=current_mulitnum+1
                        
                        elif r==8:
                            index_writetext.append([current_writetext])
                            current_writetext=current_writetext+1
            
        # 方形复选框
        if len(checkbox_id) != 0:
            ckb_center = []
            for n, p in enumerate(checkbox_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                ckb_center.append(temp)
            ckb_center = np.array(ckb_center)

            # 按行拆分
            temp_ckb_center = ckb_center.copy()
            temp_ckb_center = np.insert(temp_ckb_center,0,range(0,temp_ckb_center.shape[0]),axis=1)
            h_sort = ckb_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = ckb_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_ckb_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_ckb_center[temp_loc,:]
                row_loc = np.array(row_loc)
                row_loc = row_loc[row[:,2].argsort()]
                index.append(list(row_loc))
                temp_ckb_center = np.delete(temp_ckb_center, temp_loc, axis=0)

            index = list(filter(None, index))
            index_ckb = index

        # 圆形复选框
        if len(cirbox_id) != 0:
            cir_center = []
            for n, p in enumerate(cirbox_id):
                templet_corner = np.array(templet_corner)
                roi_corner = templet_corner[p, :]  # (h*w)
                roi_left_top, roi_right_bottom, roi_right_top, roi_left_bottom = self.findApex(roi_corner)
                temp = [(roi_left_top[0] + roi_right_bottom[0]) / 2, (roi_left_top[1] + roi_right_bottom[1]) / 2]
                cir_center.append(temp)
            cir_center = np.array(cir_center)

            # 按行拆分
            temp_cir_center = cir_center.copy()
            temp_cir_center = np.insert(temp_cir_center,0,range(0,temp_cir_center.shape[0]),axis=1)
            h_sort = cir_center[:,0].argsort()
            index = []
            for i in h_sort:
                temp_c = cir_center[i,:]
                temp_loc = []
                row_loc = []
                for j,k in enumerate(temp_cir_center):
                    if abs(temp_c[0]-k[1]) <= 30:
                        temp_loc.append(j)
                        row_loc.append(int(k[0]))
                row = temp_cir_center[temp_loc,:]
                row_loc = np.array(row_loc)
                row_loc = row_loc[row[:,2].argsort()]
                index.append(list(row_loc))
                temp_cir_center = np.delete(temp_cir_center, temp_loc, axis=0)

            index = list(filter(None, index))
            index_cir = index

        return index_num,index_ckb,index_cir,index_ocr,index_mulitnum,index_writetext,index_writetextcheck,index_mixtext

    # 根据交点提取目标区域主函数
    def get_roi_set(self,imgname):

        # 模板读取
        templet_corner,templet_num,templet_vertex,number_id,checkbox_id, cirbox_id, ocr_id,mulitnum_id,writetext_id,writetextcheck_id,mixtext_id,mixtext_type,mixtext_loc,mixtext_rowloc, img_h,img_w = self.get_template_corner()
        # 拆分成字串
        index_num,index_ckb,index_cir,index_ocr,index_mulitnum,index_writetext,index_writetextcheck,index_mixtext = self.sort_string(templet_corner, number_id, checkbox_id, cirbox_id,ocr_id,mulitnum_id,writetext_id,writetextcheck_id,mixtext_id,mixtext_type,mixtext_rowloc)
        # 仿射纠偏
        rectify_img, rectify_seg = self.affine_correction(self.img,self.seg,templet_vertex,img_h,img_w)
        # 分割结果交点检测
        seg_corner = self.seg_corner_detection(rectify_seg,imgname)
        # 模板坐标系矫正     
        templet_corner_list = np.array(templet_corner)      
        x1,y1,w1,h1 = cv.boundingRect(templet_corner_list)
        seg_corner_list = np.array(seg_corner)      
        x2,y2,w2,h2 = cv.boundingRect(seg_corner_list)
        templet_corner = templet_corner*np.array([w2/w1,h2/h1]) 
        templet_corner =  templet_corner+np.array([x2-x1*(w2/w1),y2-y1*(h2/h1)]) 
        templet_corner = np.int32(np.ceil(templet_corner))
        
        
        # 交点校验
        true_corner, true_num = self.corner_check(seg_corner,templet_corner,templet_num,rectify_img,imgname)

        cropped_number = []
        cropped_checkbox = []
        cropped_cirbox = []
        cropped_ocr = []
        cropped_mulitnum = []
        cropped_writetext = []
        cropped_writetextcheck = []

        if len(number_id) != 0:
            for n, p in enumerate(number_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'num',p)
                if cropped.size!=0:
                    cropped = cv.resize(cropped, (32, 32), interpolation=cv.INTER_LINEAR)  # (w,h)
                    _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                cropped_number.append(cropped)
        
        if len(ocr_id) != 0:
            for n, p in enumerate(ocr_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'ocr',p)

                _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                cropped_ocr.append(cropped)

        if len(mulitnum_id) != 0:
            for n, p in enumerate(mulitnum_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'mulitnum',p)
                _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                cropped_mulitnum.append(cropped)

        if len(writetext_id) != 0:
            for n, p in enumerate(writetext_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'writetext',p)
                _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                cropped_writetext.append(cropped)

        if len(writetextcheck_id) != 0:
            for n, p in enumerate(writetextcheck_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'writetextcheck',p)
                _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                cv.imwrite('D:/picdesign/TabRecoPro/data/cropped_' + str(n) +'.png', cropped)
                cropped_writetextcheck.append(cropped) 

        # 混合的文本
        temid=0
        if len(mixtext_id) != 0:
            for n, p in enumerate(mixtext_id):
                # 按columnLoc进行拆分
                print(p)
                cropped_begin=0
                cropped_rowbegin=0
                # 加入行数的计算
                for a, b in enumerate(mixtext_rowloc[n]):  
                    cropped_begin=0
                        
                    for m, r in enumerate(mixtext_type[n]):
                        if  r!=0:     
                            cropped = self.extract_roi_mix(true_corner, rectify_img, cropped_begin,cropped_rowbegin,mixtext_loc[n][m],b,r,p,temid)
                            
                        cropped_begin=cropped_begin+mixtext_loc[n][m]
                        temid=temid+1
                        # 1、手写单数,4、印刷体ocr，7、手写多数字，8、手写汉字
                        if  r==1: 
                            cropped = cv.resize(cropped, (32, 32), interpolation=cv.INTER_LINEAR)  # (w,h)
                            _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                            cropped_number.append(cropped)
                        elif r==4:
                            # 使用投影法过滤较小的干扰线
                            if len(mixtext_rowloc[n])>1:
                                vv_list=self.get_rowList(cropped)

                                if len(vv_list)==2:
                                    print(max(vv_list))
                                    maxlist=max(vv_list)
                                    if len(maxlist)>=10:
                                        img_hor=cropped[maxlist[0]:maxlist[-1]]
                                         
                                        gray = cv.cvtColor(img_hor, cv.COLOR_GRAY2BGR)
                                        img_hor = self.patch_ocr_preprocessing(gray)
                                        cropped_ocr.append(img_hor)
                                    else:
                                        cropped_ocr.append(cropped)
                                else:
                                    _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                                    cropped_ocr.append(cropped)
                            else:
                                    cropped_ocr.append(cropped)
                        elif r==7:
                            _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                            cropped_mulitnum.append(cropped)
                        elif r==8:
                            _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                            cropped_writetext.append(cropped)
                    cropped_rowbegin=cropped_rowbegin+b
                    #
        if len(checkbox_id) != 0:
            for n, p in enumerate(checkbox_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'ckb',p)
                cropped = cv.resize(cropped, (28, 28), interpolation=cv.INTER_LINEAR)  # (w,h)
                _, cropped = cv.threshold(cropped[:,:,0], 254, 255, cv.THRESH_BINARY)
                cv.imwrite('cropped_' + str(n) +'.png', cropped)
                cropped_checkbox.append(cropped)

        if len(cirbox_id) != 0:
            for n, p in enumerate(cirbox_id):
                cropped = self.extract_roi(true_corner, rectify_img, 'cir',p)
                cropped = cv.resize(cropped, (28, 28), interpolation=cv.INTER_LINEAR)  # (w,h)
                _, cropped = cv.threshold(cropped, 254, 255, cv.THRESH_BINARY)
                cropped_cirbox.append(cropped)

        # 返回值包括cropped_number: 一张图像中的数字图像块集合
        # cropped_checkbox: 一张图像中的方形checkbox图像块集合
        # cropped_cirbox: 一张图像中的圆形checkbox图像块集合
        # index_num、index_ckb、index_cir: 表示数字串、方形checkbox、圆形checkbox索引
        return cropped_number,cropped_checkbox,cropped_cirbox,cropped_ocr,cropped_mulitnum,cropped_writetext,cropped_writetextcheck,index_num,index_ckb,index_cir,index_ocr,index_mulitnum,index_writetext,index_writetextcheck

#------------------------根据交点提取目标区域-----------------------#

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        # print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    # print('Create path - %s' % dir_path)


#去除上下左右多余的空白
def delBlank(img_bgr):
    if len(img_bgr.shape) > 2:
        img_bgr = img_bgr[:, :, 0]
    #二值化
    t,binary=cv.threshold(img_bgr,127,255,cv.THRESH_BINARY)
    rows,cols=binary.shape

    '''
    水平投影从左向右投射，计算每一行的黑色像素总数
    '''
    
    hor_list=[0]*rows
    for i in range(rows):
        for j in range(cols):
            #统计每一行的黑色像素总数
            if binary.item(i,j)==0:
                hor_list[i]=hor_list[i]+1

    hor_list[rows-1] = 0

    hei = 0
    for index,i in enumerate(hor_list):
        if  i>0 :
            hei += 1
        else:
            if hei > 0 and hei < 8.0:
                hor_list[index-hei:index] = [0]
            hei = 0    

    # '''
    # 对hor_list中的元素进行筛选，可以去除一些噪点
    # '''
    # hor_arr=np.array(hor_list)
    # hor_arr[np.where(hor_arr<10)]=0
    # hor_list=hor_arr.tolist()
    start = -1
    end = -1
    for index,i in enumerate(hor_list):
        if  i>0 :
            end = index
            if start == -1:
                start = index
    stepx = 2
    if start-stepx<0:
        start = 0
        stepx = 0
    stepy = 2
    if end+stepy>rows:
        end = rows
        stepy = 0
    roiImg = img_bgr[start-stepx:end+stepy,:].copy() 
    
    '''
    垂直投影从上向下投射，计算每一行的黑色像素总数
    '''
    rows1,cols1=roiImg.shape
    ver_list1=[0]*cols1
    for j in range(cols1):
        for i in range(rows1):
            #统计每一行的黑色像素总数
            if roiImg.item(i,j)==0:
                ver_list1[j]=ver_list1[j]+1
    
    ver_list1[cols1-1] = 0
    hei = 0
    for index,i in enumerate(ver_list1):
        if  i>0 :
            hei += 1
        else:
            if hei > 0 and hei < 8.0:
                ver_list1[index-hei:index] = [0]
            hei = 0   


    start = -1
    end = -1
    for index,i in enumerate(ver_list1):
        if  i>0 :
            end = index
            if start == -1:
                start = index
    stepx = 2
    if start-stepx<0:
        start = 0
        stepx = 0
    stepy = 2
    if end+stepy>cols1:
        end = cols1
        stepy = 0
         
    roiImg = roiImg[:,start-stepx:end+stepy].copy() 
    
    return roiImg

#找到最大区域并填充
def select_max_region(mask):
    
    mask = cv.bitwise_not(mask)
    k = np.ones((6, 6), np.uint8)
    open = cv.morphologyEx(mask, cv.MORPH_CLOSE, k,1)
    t,binary=cv.threshold(open,127,255,cv.THRESH_OTSU+cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    area=[]
    if len(contours) > 0:
        for j in range(len(contours)):
            area.append(cv.contourArea(contours[j]))
        max_idx=np.argmax(area)
        max_area=cv.contourArea(contours[max_idx])
        x,y,w,h = cv.boundingRect(contours[max_idx])
        roiImg = mask[y:y+h,x:x+w].copy()
        mask[:,:] = 0
        mask[y:y+h,x:x+w] = roiImg
        # for k in range(len(contours)):
        #     if k != max_idx:
        #         cv.fillPoly(mask,[contours[k]],0)
    mask = cv.bitwise_not(mask)
    return mask

if __name__=='__main__':

    start = time.time()
    reco = recognition()
    end = time.time()
    name_total = []
    cornerdiff_total = []
    print('初始化时间：',end - start)

    # 批处理

    img_list = []
    filePath = 'D:/picdesign/dataset/total_patch/gexian-01'  
    filenames = os.listdir(filePath)
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        print(os.path.splitext(filename)[0])
        last_two = os.path.splitext(filename)[0][-2:]
        if (ext == '.tif' or ext == '.jpg')  :
            img_list.append(filePath + '/' + filename)

    start_all = time.time()
    save_path = 'D:/picdesign/dataset/total_patch/1206n/'
    correct=0
    i=0
    for img_path in img_list:
        imgname = img_path.split('/')[-1].split('.')[0]#img_path.split('/')[-2] + '_' + 
        
        realcls=imgname.split('_')[0]
        start = time.time()


        form_set = cv.imread(img_path)
        form_set1 = cv.resize(form_set,(960,960),interpolation=cv.INTER_AREA)
        form1 = []
        form1.append(form_set1)
        maxvalue,cls,form_seg=reco.formSeg_main(np.array(form1))


        form_seg =form_seg.squeeze().astype('uint8')
        saveFilePath = save_path + imgname+'.png' #+ '/seg/' + imgname+'_seg.png'
        cv.imencode('.png', form_seg)[1].tofile(saveFilePath)
        print(imgname.split('.')[0] + '表单类别结果：',cls)


        patch_catch1 = patch_catch(templet_xml_path = 'J:/bk/TabRecoPro/TabRecoPro/templet/xml/templet_' + str(cls) + '.xml',img = form_set,seg = form_seg)


        try:
            # 待识别图像块提取
            cropped_number,cropped_checkbox,cropped_cirbox,cropped_ocr,cropped_mulitnum,cropped_writetext,cropped_writetextcheck,index_num,index_ckb,index_cir,index_ocr,index_mulitnum,index_writetext,index_writetextcheck = patch_catch1.get_roi_set(imgname)

            end = time.time()

            print('提取花费时间：',end - start)

            start = time.time()
            
            # ocr图像块识别
            if len(cropped_ocr) != 0:
                # ocrCls_result = np.array(reco.ocrCls_main(np.array(cropped_ocr)))
                rm_mkdir(save_path + imgname + '/ocr/')
                for g, i in enumerate(index_ocr):
                    # for q, j in enumerate(i):
                    cropped_t = cropped_ocr[g]
                    
                    # cropped_t = delBlank(cropped_t)
                    # cv.imwrite(save_path + imgname + '/ocr/' + '%s_%s_%d_%d_%s.png' % (
                    #             imgname, 'ocr', g, 0, str(int(1))), cropped_t)
                    saveFilePath = save_path + imgname + '/ocr/' + '%s_%s_%d_%d_%s.png' % (
                                imgname, 'ocr', g, 0, str(int(1)))
                    cv.imencode('.png', cropped_t)[1].tofile(saveFilePath)
                    ocrCls_result = (reco.ocrCls_main(save_path + imgname + '/ocr/' + '%s_%s_%d_%d_%s.png' % (
                                imgname, 'ocr', g, 0, str(int(1)))))
                    name_total.append(save_path + imgname + '/ocr/' + '%s_%s_%d_%d_%s.png' % (
                                imgname, 'ocr', g, 0, str(int(1))))
                    cornerdiff_total.append(ocrCls_result)            
                    print(save_path + imgname + '/ocr/' + '%s_%s_%d_%d_%s.png' % (
                                    imgname, 'ocr', g, 0, str(int(1)))+"   "+ocrCls_result)

            # 多数字手写图像块识别
            if len(cropped_mulitnum) != 0:
                mulitnumCls_result = np.array(reco.mulitnumCls_main(np.array(cropped_mulitnum)))
                rm_mkdir(save_path + imgname + '/mulitnum/')
                for g, i in enumerate(index_mulitnum):
                    for q, j in enumerate(i):
                        cropped_t = cropped_mulitnum[j]
                        # cv.imwrite(save_path + imgname + '/mulitnum/' + '%s_%s_%d_%d_%s.png' % (
                        #             imgname, 'mulitnum', g, q, str(int(2))), cropped_t)
                        saveFilePath = save_path + imgname + '/mulitnum/' + '%s_%s_%d_%d_%s.png' % (
                                    imgname, 'mulitnum', g, q, str(int(2)))
                        # cropped_t = delBlank(cropped_t)
                        cv.imencode('.png', cropped_t)[1].tofile(saveFilePath)

            # 手写汉字图像块识别
            if len(cropped_writetext) != 0:
                writetextCls_result = np.array(reco.writetextCls_main(np.array(cropped_writetext)))
                rm_mkdir(save_path + imgname + '/writetext/')
                for g, i in enumerate(index_writetext):
                    for q, j in enumerate(i):
                        cropped_t = cropped_writetext[j]
                        # cv.imwrite(save_path + imgname + '/writetext/' + '%s_%s_%d_%d_%s.png' % (
                        #             imgname, 'writetext', g, q, str(int(3))), cropped_t)
                        saveFilePath = save_path + imgname + '/writetext/' + '%s_%s_%d_%d_%s.png' % (
                                    imgname, 'writetext', g, q, str(int(3)))
                        cv.imencode('.png', cropped_t)[1].tofile(saveFilePath)

            # 新带钩数字图像块识别
            if len(cropped_writetextcheck) != 0:
                writetextcheck_result = np.array(reco.writetextcheck_main(cropped_writetextcheck))
                rm_mkdir(save_path + imgname + '/writetextcheck/')
                for g, i in enumerate(index_writetextcheck):
                    for q, j in enumerate(i):
                        cropped_t = cropped_writetextcheck[j]
                        # cv.imwrite(save_path + imgname + '/writetext/' + '%s_%s_%d_%d_%s.png' % (
                        #             imgname, 'writetext', g, q, str(int(3))), cropped_t)
                        saveFilePath = save_path + imgname + '/writetextcheck/' + '%s_%s_%d_%d_%s.png' % (
                                    imgname, 'writetextcheck', g, q, (writetextcheck_result[j]))
                        cv.imencode('.png', cropped_t)[1].tofile(saveFilePath)

            # 方形checkbox图像块识别
            if len(cropped_checkbox) != 0:
                ckbCls_result = np.array(reco.ckbCls_main(np.array(cropped_checkbox)))
                rm_mkdir(save_path + imgname + '/ckb/')
                for g, i in enumerate(index_ckb):
                    for q, j in enumerate(i):
                        cropped_t = cropped_checkbox[j]
                        # cv.imwrite(save_path + imgname + '/ckb/' + '%s_%s_%d_%d_%s.png' % (
                        #             imgname, 'ckb', g, q, str(int(ckbCls_result[j]))), cropped_t)                    
                        saveFilePath = save_path + imgname + '/ckb/' + '%s_%s_%d_%d_%s.png' % (
                                    imgname, 'ckb', g, q, str(int(ckbCls_result[j])))
                        cv.imencode('.png', cropped_t)[1].tofile(saveFilePath)
            # 方形checkbox图像块识别
            if len(cropped_cirbox) != 0:
                cirCls_result = np.array(reco.ckbCls_main(np.array(cropped_cirbox)))
                if len(cropped_checkbox) == 0:
                    rm_mkdir(save_path + imgname + '/ckb/')
                for g, i in enumerate(index_cir):
                    for q, j in enumerate(i):
                        cropped_t = cropped_cirbox[j]
                        # cv.imwrite(save_path + imgname + '/ckb/' + '%s_%s_%d_%d_%s.png' % (
                        #             imgname, 'cir', g, q, str(int(cirCls_result[j]))), cropped_t)
                        saveFilePath = save_path + imgname + '/ckb/' + '%s_%s_%d_%d_%s.png' % (
                                    imgname, 'cir', g, q, str(int(cirCls_result[j])))
                        cv.imencode('.png', cropped_t)[1].tofile(saveFilePath)

            end = time.time()

            print('识别花费时间：',end - start)
        except Exception as e:
            print('捕获到异常',e) 
            print('文件', e.__traceback__.tb_frame.f_globals['__file__'])
               
            # delDir(savePath)   
            print('行号', e.__traceback__.tb_lineno)
            print('表格提取异常:'+img_path)


    end_all = time.time()
    print('总耗时：',end_all-start)











