import pandas as pd
import numpy as np
import os
import cv2
def graying(img):
    img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    img = thresh
    return thresh
def splitImg(img):
    #img = graying(img)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        #print([x,y,w,h])
    img = img[y:y+h,x:x+w]
    return img
def standard(img):
    w,h = img.shape[1],img.shape[0]
    w_i,h_i=[],[]
    n_grids = 10
    # 计算每一个单元格w与h的大小
    '''if w%5 != 0:
        if w%5 > 5:
            for i in range(4):w_i.append(w//5+1)
            w_i.append(w_i[0]*5-w+w//5)
        else:
            for i in range(4):w_i.append(w//5)
            w_i.append(w-w_i[0]*4)
    else:
        for i in range(5):w_i.append(w//5)
    
    if h%5 != 0:
        if h%5 > 5:
            for i in range(4):h_i.append(h//5+1)
            h_i.append(h_i[0]*5-h+h//5)
        else:
            for i in range(4):h_i.append(h//5)
            h_i.append(h-h_i[0]*4)
    else:
        for i in range(5):h_i.append(h//5)'''
    for i in range(n_grids):w_i.append(w//n_grids)
    for i in range(n_grids):h_i.append(w//n_grids)
    # 归一化
    loss = 0.2
    w_end,h_end = 0,0
    arr_matrix = []
    for i in range(n_grids):
        col = w_end
        w_end += w_i[i]
        h_end = 0
        for j in range(n_grids):
            count = 0
            count_a = 0
            row = h_end
            col = w_end-w_i[i]
            h_end += h_i[j]
            while col<w_end:
                row = h_end-h_i[j]
                while row<h_end:
                    if img[row,col] > 150:
                        
                        count +=1
                    count_a +=1
                    row +=1
                col +=1
            if count/count_a > loss:
                arr_matrix.append(1)
            else:
                arr_matrix.append(0)
            
    matrix = np.array(arr_matrix).reshape(n_grids,n_grids).T
    return matrix
def getTrain():
    local = './data/train-images/'
    if_train = 1
    if os.path.exists(r"./data/output.csv"):
        input_f_csv = pd.read_csv("./data/output.csv",index_col=0)
        #print(input_f_csv.shape)
        #print(input_f_csv)
        if input_f_csv.shape[0] == 10 and input_f_csv.shape[1] == 100:
            if_train = 0
            std_library = input_f_csv
        else:
            os.remove(r"./data/output.csv")
    if if_train:
        std_library = pd.DataFrame(columns=range(100))#1-->100
        for n in range(10):#1-->10
            arr_ =[]
            for m in range(100):#1-->100
                img = cv2.imread("%s%s_%s.bmp"%(local,n,m))
                img = graying(img)
                img = splitImg(img)
                img = cv2.resize(img,(50,50))
                ret,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
                std_matrix = standard(img)
                str_std = ''
                for i in std_matrix.tolist():str_std = "%s,%s"%(str_std,",".join(map(str,i)))
                arr_.append(str_std[1:])
                # print(std_matrix)
            std_library.loc[n] = arr_

        #print(std_library)
        std_library.to_csv("./data/output.csv")
        #imshow(img)
        print("Start from creating.")
    else:
        print("Start from CSV file.")
    
    return std_library
def main():
    local_test = './data/test-images/'
    dic_output = {}
    std_library = getTrain()
    for n in range(10):#1-->10
        for m in range(20):#1-->20
            img = cv2.imread("%s%s_%s.bmp"%(local_test,n,m))
            img = graying(img)
            img = splitImg(img)
            img = cv2.resize(img,(50,50))
            ret,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
            train_matrix = standard(img)
            #print(train_matrix)
            for row in std_library.itertuples():
                hit = 0
                for i in range(1,101):#2-->101
                    now_str = getattr(row,"_%s"%i)
                    now_matrix = np.array(list(map(int,now_str.split(',')))).reshape(10,10)
                    sub = now_matrix-train_matrix
                    if np.sum(sub==0)/100 > hit: 
                        hit = np.sum(sub==0)/100
                    if hit == 1.0:
                        break
                if hit == 1.0:
                    str_name = "%s_%s"%(n,m)
                    dic_output[str_name] = row.Index
                    #print(row.Index)
                    break
    print(dic_output)
if __name__ == "__main__":
    main()